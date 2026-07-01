"""Unit tests for loss functions in cellmap_segmentation_challenge.utils.loss"""

import pytest
import torch
import torch.nn.functional as F

from cellmap_segmentation_challenge.utils.loss import (
    CellMapDynamicWeightedCrossEntropyLoss,
    CellMapForegroundCEBackgroundRejectionLoss,
    CellMapLossWrapper,
)


class TestCellMapLossWrapper:
    """Tests for CellMapLossWrapper class"""

    def test_init_with_mse_loss(self):
        """Test initialization with MSE loss"""
        loss_wrapper = CellMapLossWrapper(torch.nn.MSELoss)
        assert isinstance(loss_wrapper.loss_fn, torch.nn.MSELoss)
        assert loss_wrapper.kwargs["reduction"] == "none"

    def test_init_with_bce_loss(self):
        """Test initialization with BCE loss"""
        loss_wrapper = CellMapLossWrapper(torch.nn.BCELoss)
        assert isinstance(loss_wrapper.loss_fn, torch.nn.BCELoss)

    def test_calc_loss_no_nans(self):
        """Test calc_loss with no NaN values"""
        loss_wrapper = CellMapLossWrapper(torch.nn.MSELoss)
        outputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        targets = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        loss = loss_wrapper.calc_loss(outputs, targets)

        # With identical outputs and targets, MSE should be 0
        assert torch.allclose(loss, torch.tensor(0.0))

    def test_calc_loss_with_nans(self):
        """Test calc_loss with NaN values in targets"""
        loss_wrapper = CellMapLossWrapper(torch.nn.MSELoss)
        outputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        targets = torch.tensor([[1.0, float("nan")], [3.0, 4.0]])

        loss = loss_wrapper.calc_loss(outputs, targets)

        # Loss should be computed only for non-NaN values
        # Expected loss = ((1-1)^2 + (3-3)^2 + (4-4)^2) / 3 = 0
        assert torch.allclose(loss, torch.tensor(0.0))

    def test_calc_loss_all_nans(self):
        """Test calc_loss when all targets are NaN"""
        loss_wrapper = CellMapLossWrapper(torch.nn.MSELoss)
        outputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        targets = torch.tensor(
            [[float("nan"), float("nan")], [float("nan"), float("nan")]]
        )

        loss = loss_wrapper.calc_loss(outputs, targets)

        # When all targets are NaN, the loss is 0 (no valid pixels to compute loss on)
        assert torch.allclose(loss, torch.tensor(0.0)) or torch.isnan(loss)

    def test_forward_tensor_inputs(self):
        """Test forward with tensor inputs"""
        loss_wrapper = CellMapLossWrapper(torch.nn.MSELoss)
        outputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        targets = torch.tensor([[1.5, 2.5], [3.5, 4.5]])

        loss = loss_wrapper.forward(outputs, targets)

        # MSE = ((1-1.5)^2 + (2-2.5)^2 + (3-3.5)^2 + (4-4.5)^2) / 4 = 0.25
        assert torch.allclose(loss, torch.tensor(0.25), atol=1e-6)

    def test_forward_dict_inputs_matching_dicts(self):
        """Test forward with matching dictionary inputs"""
        loss_wrapper = CellMapLossWrapper(torch.nn.MSELoss)
        outputs = {
            "class1": torch.tensor([[1.0, 2.0]]),
            "class2": torch.tensor([[3.0, 4.0]]),
        }
        targets = {
            "class1": torch.tensor([[1.0, 2.0]]),
            "class2": torch.tensor([[3.0, 4.0]]),
        }

        loss = loss_wrapper.forward(outputs, targets)

        # Perfect match, loss should be 0
        assert torch.allclose(loss, torch.tensor(0.0))

    def test_forward_dict_inputs_with_nans(self):
        """Test forward with dictionary inputs containing NaN values"""
        loss_wrapper = CellMapLossWrapper(torch.nn.MSELoss)
        outputs = {
            "class1": torch.tensor([[1.0, 2.0]]),
            "class2": torch.tensor([[3.0, 4.0]]),
        }
        targets = {
            "class1": torch.tensor([[1.0, float("nan")]]),
            "class2": torch.tensor([[3.0, 4.0]]),
        }

        loss = loss_wrapper.forward(outputs, targets)

        # Loss from class1: only first element counted (0)
        # Loss from class2: both elements (0)
        # Average of 0 and 0 = 0
        assert torch.allclose(loss, torch.tensor(0.0))

    def test_forward_dict_targets_list_outputs(self):
        """Test forward with dict targets and list/tuple outputs"""
        loss_wrapper = CellMapLossWrapper(torch.nn.MSELoss)
        outputs = [torch.tensor([[1.0, 2.0]]), torch.tensor([[3.0, 4.0]])]
        targets = {
            "class1": torch.tensor([[1.0, 2.0]]),
            "class2": torch.tensor([[3.0, 4.0]]),
        }

        loss = loss_wrapper.forward(outputs, targets)

        # Perfect match, loss should be 0
        assert torch.allclose(loss, torch.tensor(0.0))

    def test_bce_loss_with_nans(self):
        """Test with BCE loss and NaN values"""
        loss_wrapper = CellMapLossWrapper(torch.nn.BCELoss)
        outputs = torch.tensor([[0.5, 0.8], [0.3, 0.9]])
        targets = torch.tensor([[1.0, float("nan")], [0.0, 1.0]])

        loss = loss_wrapper.calc_loss(outputs, targets)

        # Loss should be computed only for non-NaN values
        assert not torch.isnan(loss)
        assert loss >= 0  # BCE loss is always non-negative


class TestCellMapDynamicWeightedCrossEntropyLoss:
    def test_patch_weights_balance_active_classes(self):
        target_indices = torch.tensor([[[0, 0, 0, 0], [0, 0, 1, 1]]])
        targets = F.one_hot(target_indices, num_classes=4).movedim(-1, 1).float()
        outputs = torch.randn(1, 4, 2, 4, requires_grad=True)

        loss = CellMapDynamicWeightedCrossEntropyLoss()(outputs, targets)
        loss.backward()

        assert torch.isfinite(loss)
        assert torch.isfinite(outputs.grad).all()

    def test_max_class_weight_caps_extreme_inverse_frequency(self):
        target_indices = torch.cat(
            [torch.zeros(99, dtype=torch.long), torch.ones(1, dtype=torch.long)]
        ).reshape(1, 10, 10)
        targets = F.one_hot(target_indices, num_classes=4).movedim(-1, 1).float()
        outputs = torch.zeros(1, 4, 10, 10, requires_grad=True)

        loss = CellMapDynamicWeightedCrossEntropyLoss(max_class_weight=25.0)(
            outputs,
            targets,
        )

        assert torch.allclose(loss, torch.log(torch.tensor(4.0)), atol=1e-6)

    def test_min_class_weight_prevents_majority_downweighting(self):
        target_indices = torch.cat(
            [torch.zeros(75, dtype=torch.long), torch.ones(25, dtype=torch.long)]
        ).reshape(1, 10, 10)
        targets = F.one_hot(target_indices, num_classes=4).movedim(-1, 1).float()
        outputs = torch.zeros(1, 4, 10, 10, requires_grad=True)

        loss = CellMapDynamicWeightedCrossEntropyLoss(
            min_class_weight=1.0,
            max_class_weight=100.0,
        )(outputs, targets)

        assert torch.allclose(loss, torch.log(torch.tensor(4.0)), atol=1e-6)


class TestCellMapForegroundCEBackgroundRejectionLoss:
    def test_foreground_uses_ordinary_cross_entropy(self):
        outputs = torch.tensor(
            [[[[2.0, 0.0]], [[0.0, 2.0]]]],
            requires_grad=True,
        )
        targets = torch.zeros(1, 3, 1, 2)
        targets[0, 0, 0, 0] = 1
        targets[0, 1, 0, 1] = 1

        loss = CellMapForegroundCEBackgroundRejectionLoss(
            background_penalty_weight=0.0
        )(outputs, targets)
        expected = F.cross_entropy(
            outputs,
            torch.tensor([[[0, 1]]]),
        )

        assert torch.allclose(loss, expected)

    def test_background_penalizes_only_confidence_above_threshold(self):
        targets = torch.zeros(1, 3, 1, 2)
        targets[:, -1] = 1
        outputs = torch.tensor(
            [[[[0.0, 4.0]], [[0.0, 0.0]]]],
            requires_grad=True,
        )

        loss = CellMapForegroundCEBackgroundRejectionLoss(
            confidence_threshold=0.6,
            background_penalty_weight=1.0,
            penalty_power=2.0,
        )(outputs, targets)
        probabilities = F.softmax(outputs, dim=1).amax(dim=1)
        expected = F.relu(probabilities - 0.6).pow(2).mean()

        assert torch.allclose(loss, expected)
        loss.backward()
        assert torch.isfinite(outputs.grad).all()

    def test_requires_one_extra_background_channel(self):
        outputs = torch.zeros(1, 2, 2, 2)
        targets = torch.zeros(1, 2, 2, 2)

        with pytest.raises(ValueError, match="plus one bg channel"):
            CellMapForegroundCEBackgroundRejectionLoss()(outputs, targets)
