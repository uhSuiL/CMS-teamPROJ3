import torch
import torch.nn.functional as F


class CellMapLossWrapper(torch.nn.modules.loss._Loss):
    """
    Wrapper for any PyTorch loss function that is applied to the output of a model and the target.

    Because the target can contain NaN values, the loss function is applied only to the non-NaN values.
    This is done by multiplying the loss by a mask that is 1 where the target is not NaN and 0 where the target is NaN.
    The loss is then averaged across the non-NaN values.

    Parameters
    ----------
    loss_fn : torch.nn.modules.loss._Loss or torch.nn.modules.loss._WeightedLoss
        The loss function to apply to the output and target.
    **kwargs
        Keyword arguments to pass to the loss function.
    """

    def __init__(
        self,
        loss_fn: torch.nn.modules.loss._Loss | torch.nn.modules.loss._WeightedLoss,
        **kwargs,
    ):
        super().__init__()
        self.kwargs = kwargs
        self.kwargs["reduction"] = "none"
        self.loss_fn = loss_fn(**self.kwargs)

    def calc_loss(self, outputs: torch.Tensor, target: torch.Tensor):
        loss = self.loss_fn(outputs, target.nan_to_num(0))
        loss = (loss * target.isnan().logical_not()).nanmean()
        return loss

    def forward(
        self,
        outputs: dict | torch.Tensor,
        targets: dict | torch.Tensor,
    ):
        if isinstance(targets, dict):
            loss = 0
            if isinstance(outputs, dict):
                for key, target in targets.items():
                    loss += self.calc_loss(outputs[key], target)
            else:
                # Assumes outputs is a list or tuple of tensors aligned with targets
                for i, target in enumerate(targets.values()):
                    loss += self.calc_loss(outputs[i], target)
            loss /= len(targets)
        else:
            loss = self.calc_loss(outputs, targets)  # type: ignore
        return loss


class CellMapCrossEntropyLoss(torch.nn.Module):
    """
    Multi-class cross entropy for CellMap-style one-channel-per-class targets.

    Converts a target tensor of shape (B, C, ...) into class indices of shape
    (B, ...) using argmax over the class dimension, then applies CE to raw
    logits of shape (B, C, ...).
    """

    def __init__(self, ignore_index: int = -100, **kwargs):
        super().__init__()
        self.ignore_index = ignore_index
        self.kwargs = kwargs

    def _target_to_indices(self, targets: torch.Tensor) -> torch.Tensor:
        valid = targets.isnan().logical_not().any(dim=1)
        target_indices = targets.nan_to_num(0).argmax(dim=1).long()
        return target_indices.masked_fill(valid.logical_not(), self.ignore_index)

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor):
        target_indices = self._target_to_indices(targets)
        return F.cross_entropy(
            outputs,
            target_indices,
            ignore_index=self.ignore_index,
            **self.kwargs,
        )


class CellMapDynamicWeightedCrossEntropyLoss(CellMapCrossEntropyLoss):
    """Patch-wise inverse-frequency weighted cross-entropy.

    For each patch independently, an active class ``c`` receives weight
    ``N / (C_active * n_c)``, where ``N`` is the number of valid voxels,
    ``C_active`` is the number of classes present in the patch, and ``n_c`` is
    the class voxel count. Missing classes receive weight zero. Optional lower
    and upper bounds control majority-class downweighting and rare-class
    amplification.
    """

    def __init__(
        self,
        min_class_weight: float | None = None,
        max_class_weight: float | None = 25.0,
        ignore_index: int = -100,
        **kwargs,
    ):
        super().__init__(ignore_index=ignore_index, **kwargs)
        if min_class_weight is not None and min_class_weight <= 0:
            raise ValueError("min_class_weight must be positive or None")
        if max_class_weight is not None and max_class_weight <= 0:
            raise ValueError("max_class_weight must be positive or None")
        if (
            min_class_weight is not None
            and max_class_weight is not None
            and min_class_weight > max_class_weight
        ):
            raise ValueError(
                "min_class_weight cannot be greater than max_class_weight"
            )
        self.min_class_weight = min_class_weight
        self.max_class_weight = max_class_weight

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor):
        target_indices = self._target_to_indices(targets)
        voxel_losses = F.cross_entropy(
            outputs,
            target_indices,
            ignore_index=self.ignore_index,
            reduction="none",
            **self.kwargs,
        )

        patch_losses = []
        num_classes = outputs.shape[1]
        for patch_index in range(outputs.shape[0]):
            patch_targets = target_indices[patch_index]
            valid = patch_targets != self.ignore_index
            valid_count = valid.sum()
            if valid_count == 0:
                patch_losses.append(outputs[patch_index].sum() * 0.0)
                continue

            class_counts = torch.bincount(
                patch_targets[valid],
                minlength=num_classes,
            ).to(device=outputs.device, dtype=outputs.dtype)
            active = class_counts > 0
            active_count = active.sum().to(dtype=outputs.dtype)

            class_weights = torch.zeros_like(class_counts)
            class_weights[active] = valid_count.to(outputs.dtype) / (
                active_count * class_counts[active]
            )
            if self.min_class_weight is not None:
                class_weights[active] = class_weights[active].clamp(
                    min=self.min_class_weight
                )
            if self.max_class_weight is not None:
                class_weights[active] = class_weights[active].clamp(
                    max=self.max_class_weight
                )

            voxel_weights = class_weights[patch_targets[valid]]
            weighted_loss = voxel_losses[patch_index][valid] * voxel_weights
            patch_losses.append(weighted_loss.sum() / voxel_weights.sum())

        return torch.stack(patch_losses).mean()


class CellMapForegroundCEBackgroundRejectionLoss(torch.nn.Module):
    """Foreground CE with confidence rejection on a separate background mask.

    The model predicts ``C`` foreground classes while the target contains
    ``C + 1`` channels. The final target channel is a background mask:

    - foreground voxels use ordinary multi-class cross-entropy;
    - background voxels do not enter CE;
    - background voxels are penalized when the largest foreground softmax
      probability exceeds ``confidence_threshold``.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.5,
        background_penalty_weight: float = 1.0,
        penalty_power: float = 2.0,
        **kwargs,
    ):
        super().__init__()
        if not 0.0 < confidence_threshold < 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if background_penalty_weight < 0.0:
            raise ValueError("background_penalty_weight must be non-negative")
        if penalty_power <= 0.0:
            raise ValueError("penalty_power must be positive")

        self.confidence_threshold = confidence_threshold
        self.background_penalty_weight = background_penalty_weight
        self.penalty_power = penalty_power
        self.ce_kwargs = kwargs

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor):
        num_foreground_classes = outputs.shape[1]
        expected_target_channels = num_foreground_classes + 1
        if targets.shape[1] != expected_target_channels:
            raise ValueError(
                "Background-rejection loss expects one target channel per "
                f"foreground class plus one bg channel: expected "
                f"{expected_target_channels}, got {targets.shape[1]}."
            )

        foreground_targets = targets[:, :num_foreground_classes].nan_to_num(0)
        background_mask = targets[:, -1].nan_to_num(0) > 0.5
        foreground_mask = foreground_targets.sum(dim=1) > 0.5

        target_indices = foreground_targets.argmax(dim=1)
        if foreground_mask.any():
            voxel_ce = F.cross_entropy(
                outputs,
                target_indices,
                reduction="none",
                **self.ce_kwargs,
            )
            foreground_ce = voxel_ce[foreground_mask].mean()
        else:
            foreground_ce = outputs.sum() * 0.0

        if background_mask.any() and self.background_penalty_weight > 0.0:
            max_foreground_probability = F.softmax(outputs, dim=1).amax(dim=1)
            excess_confidence = F.relu(
                max_foreground_probability - self.confidence_threshold
            )
            background_penalty = (
                excess_confidence[background_mask].pow(self.penalty_power).mean()
            )
        else:
            background_penalty = outputs.sum() * 0.0

        return (
            foreground_ce
            + self.background_penalty_weight * background_penalty
        )


class CellMapDiceCELoss(CellMapCrossEntropyLoss):
    """Combined Dice + CE loss for mutually exclusive CellMap labels."""

    def __init__(
        self,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        dice_smooth: float = 1.0,
        include_background: bool = True,
        class_weights: list[float] | tuple[float, ...] | torch.Tensor | None = None,
        ignore_index: int = -100,
        **kwargs,
    ):
        super().__init__(ignore_index=ignore_index, **kwargs)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.dice_smooth = dice_smooth
        self.include_background = include_background
        self.class_weights = (
            None if class_weights is None else torch.as_tensor(class_weights, dtype=torch.float32)
        )

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor):
        target_indices = self._target_to_indices(targets)
        weight = (
            None
            if self.class_weights is None
            else self.class_weights.to(device=outputs.device, dtype=outputs.dtype)
        )
        ce_loss = F.cross_entropy(
            outputs,
            target_indices,
            weight=weight,
            ignore_index=self.ignore_index,
            **self.kwargs,
        )

        valid = target_indices != self.ignore_index
        safe_target = target_indices.masked_fill(valid.logical_not(), 0)
        target_one_hot = F.one_hot(
            safe_target, num_classes=outputs.shape[1]
        ).movedim(-1, 1)
        target_one_hot = target_one_hot.to(dtype=outputs.dtype)
        probs = F.softmax(outputs, dim=1)

        valid = valid.unsqueeze(1)
        probs = probs * valid
        target_one_hot = target_one_hot * valid

        if not self.include_background and outputs.shape[1] > 1:
            probs = probs[:, 1:]
            target_one_hot = target_one_hot[:, 1:]

        reduce_dims = tuple(range(2, outputs.ndim))
        intersection = (probs * target_one_hot).sum(dim=reduce_dims)
        denominator = probs.sum(dim=reduce_dims) + target_one_hot.sum(dim=reduce_dims)
        dice_score = (2 * intersection + self.dice_smooth) / (
            denominator + self.dice_smooth
        )
        dice_loss = 1 - dice_score.mean()

        return self.ce_weight * ce_loss + self.dice_weight * dice_loss


class CellMapFilteredDynamicWeightedDiceCELoss(CellMapCrossEntropyLoss):
    """Patch-filtered dynamic inverse-frequency CE + Dice loss.

    This loss is for mutually exclusive CellMap labels stored as one binary
    channel per class. It can ignore whole patches when selected classes occupy
    too much of the patch, then trains on the remaining patches with:

    ``ce_weight * dynamic_inverse_frequency_CE + dice_weight * Dice``.
    """

    def __init__(
        self,
        ce_weight: float = 0.4,
        dice_weight: float = 0.6,
        dice_smooth: float = 1.0,
        min_class_weight: float | None = None,
        max_class_weight: float | None = 100.0,
        filter_class_indices: list[int] | tuple[int, ...] | None = None,
        filter_ratio_threshold: float | None = None,
        include_background: bool = True,
        ignore_index: int = -100,
        **kwargs,
    ):
        super().__init__(ignore_index=ignore_index, **kwargs)
        if ce_weight < 0 or dice_weight < 0:
            raise ValueError("ce_weight and dice_weight must be non-negative")
        if min_class_weight is not None and min_class_weight <= 0:
            raise ValueError("min_class_weight must be positive or None")
        if max_class_weight is not None and max_class_weight <= 0:
            raise ValueError("max_class_weight must be positive or None")
        if (
            min_class_weight is not None
            and max_class_weight is not None
            and min_class_weight > max_class_weight
        ):
            raise ValueError(
                "min_class_weight cannot be greater than max_class_weight"
            )
        if filter_ratio_threshold is not None and not 0 <= filter_ratio_threshold <= 1:
            raise ValueError("filter_ratio_threshold must be between 0 and 1")

        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.dice_smooth = dice_smooth
        self.min_class_weight = min_class_weight
        self.max_class_weight = max_class_weight
        self.filter_class_indices = (
            None if filter_class_indices is None else tuple(filter_class_indices)
        )
        self.filter_ratio_threshold = filter_ratio_threshold
        self.include_background = include_background

    def _patch_is_kept(self, patch_targets: torch.Tensor, valid: torch.Tensor) -> bool:
        if self.filter_class_indices is None or self.filter_ratio_threshold is None:
            return True

        valid_count = valid.sum()
        if valid_count == 0:
            return False

        filter_mask = torch.zeros_like(valid)
        for class_index in self.filter_class_indices:
            filter_mask |= patch_targets == class_index
        filter_ratio = (filter_mask & valid).sum().to(torch.float32) / valid_count
        return bool(filter_ratio <= self.filter_ratio_threshold)

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor):
        target_indices = self._target_to_indices(targets)
        voxel_losses = F.cross_entropy(
            outputs,
            target_indices,
            ignore_index=self.ignore_index,
            reduction="none",
            **self.kwargs,
        )
        probabilities = F.softmax(outputs, dim=1)
        num_classes = outputs.shape[1]

        patch_losses = []
        for patch_index in range(outputs.shape[0]):
            patch_targets = target_indices[patch_index]
            valid = patch_targets != self.ignore_index
            if not self._patch_is_kept(patch_targets, valid):
                continue

            valid_count = valid.sum()
            if valid_count == 0:
                continue

            class_counts = torch.bincount(
                patch_targets[valid],
                minlength=num_classes,
            ).to(device=outputs.device, dtype=outputs.dtype)
            active = class_counts > 0
            active_count = active.sum().to(dtype=outputs.dtype)

            class_weights = torch.zeros_like(class_counts)
            class_weights[active] = valid_count.to(outputs.dtype) / (
                active_count * class_counts[active]
            )
            if self.min_class_weight is not None:
                class_weights[active] = class_weights[active].clamp(
                    min=self.min_class_weight
                )
            if self.max_class_weight is not None:
                class_weights[active] = class_weights[active].clamp(
                    max=self.max_class_weight
                )
            active_weight_mean = class_weights[active].mean()
            if active_weight_mean > 0:
                class_weights[active] = class_weights[active] / active_weight_mean

            voxel_weights = class_weights[patch_targets[valid]]
            ce_loss = (
                voxel_losses[patch_index][valid] * voxel_weights
            ).sum() / voxel_weights.sum()

            safe_target = patch_targets.masked_fill(valid.logical_not(), 0)
            target_one_hot = F.one_hot(
                safe_target,
                num_classes=num_classes,
            ).movedim(-1, 0)
            target_one_hot = target_one_hot.to(dtype=outputs.dtype)
            patch_probs = probabilities[patch_index]
            valid_for_dice = valid.unsqueeze(0)
            patch_probs = patch_probs * valid_for_dice
            target_one_hot = target_one_hot * valid_for_dice

            if not self.include_background and num_classes > 1:
                patch_probs = patch_probs[:-1]
                target_one_hot = target_one_hot[:-1]

            reduce_dims = tuple(range(1, patch_probs.ndim))
            intersection = (patch_probs * target_one_hot).sum(dim=reduce_dims)
            denominator = patch_probs.sum(dim=reduce_dims) + target_one_hot.sum(
                dim=reduce_dims
            )
            dice_score = (2 * intersection + self.dice_smooth) / (
                denominator + self.dice_smooth
            )
            dice_loss = 1 - dice_score.mean()

            patch_losses.append(self.ce_weight * ce_loss + self.dice_weight * dice_loss)

        if not patch_losses:
            return outputs.sum() * 0.0
        return torch.stack(patch_losses).mean()


class CellMapFocalDiceLoss(CellMapCrossEntropyLoss):
    """Combined focal + Dice loss for imbalanced mutually exclusive labels."""

    def __init__(
        self,
        alpha: list[float] | tuple[float, ...] | torch.Tensor | None = None,
        gamma: float = 1.5,
        focal_weight: float = 0.75,
        dice_weight: float = 0.25,
        dice_smooth: float = 1.0,
        include_background: bool = True,
        ignore_index: int = -100,
        **kwargs,
    ):
        super().__init__(ignore_index=ignore_index, **kwargs)
        self.alpha = None if alpha is None else torch.as_tensor(alpha, dtype=torch.float32)
        self.gamma = gamma
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight
        self.dice_smooth = dice_smooth
        self.include_background = include_background

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor):
        target_indices = self._target_to_indices(targets)
        valid = target_indices != self.ignore_index
        safe_target = target_indices.masked_fill(valid.logical_not(), 0)

        ce_loss = F.cross_entropy(
            outputs,
            target_indices,
            ignore_index=self.ignore_index,
            reduction="none",
            **self.kwargs,
        )
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt).pow(self.gamma) * ce_loss

        if self.alpha is not None:
            alpha = self.alpha.to(device=outputs.device, dtype=outputs.dtype)
            alpha_t = alpha[safe_target]
            focal_loss = alpha_t * focal_loss

        focal_loss = focal_loss[valid].mean()

        target_one_hot = F.one_hot(
            safe_target, num_classes=outputs.shape[1]
        ).movedim(-1, 1)
        target_one_hot = target_one_hot.to(dtype=outputs.dtype)
        probs = F.softmax(outputs, dim=1)

        valid = valid.unsqueeze(1)
        probs = probs * valid
        target_one_hot = target_one_hot * valid

        if not self.include_background and outputs.shape[1] > 1:
            probs = probs[:, 1:]
            target_one_hot = target_one_hot[:, 1:]

        reduce_dims = tuple(range(2, outputs.ndim))
        intersection = (probs * target_one_hot).sum(dim=reduce_dims)
        denominator = probs.sum(dim=reduce_dims) + target_one_hot.sum(dim=reduce_dims)
        dice_score = (2 * intersection + self.dice_smooth) / (
            denominator + self.dice_smooth
        )
        dice_loss = 1 - dice_score.mean()

        return self.focal_weight * focal_loss + self.dice_weight * dice_loss
