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


class CellMapDiceCELoss(CellMapCrossEntropyLoss):
    """Combined Dice + CE loss for mutually exclusive CellMap labels."""

    def __init__(
        self,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        dice_smooth: float = 1.0,
        include_background: bool = True,
        ignore_index: int = -100,
        **kwargs,
    ):
        super().__init__(ignore_index=ignore_index, **kwargs)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.dice_smooth = dice_smooth
        self.include_background = include_background

    def forward(self, outputs: torch.Tensor, targets: torch.Tensor):
        target_indices = self._target_to_indices(targets)
        ce_loss = F.cross_entropy(
            outputs,
            target_indices,
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
