from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch
import zarr

try:
    from monai.metrics import DiceMetric, SurfaceDiceMetric
except ImportError:
    DiceMetric = None
    SurfaceDiceMetric = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_DIR = PROJECT_ROOT / "data" / "predictions"
DATA_DIR = PROJECT_ROOT / "data"

CLASSES = ["endo_lum", "cyto", "endo_mem", "pm", "ecs", "bg"]
NS_DICE_DISTANCE_TOLERANCE = 1.0


@dataclass
class CropMetrics:
    dataset: str
    crop: str
    valid_voxels: int
    correct_voxels: int
    gt_counts: np.ndarray
    one_hot_dice: np.ndarray
    logits_dice: np.ndarray
    ns_dice: np.ndarray


def format_metric(value: float) -> str:
    if np.isnan(value):
        return "nan"
    return f"{value:.4f}"


def format_percent(value: float) -> str:
    if np.isnan(value):
        return "nan"
    return f"{100.0 * value:.4f}%"


def read_level_transform(group_path: Path, level_path: str = "s0") -> tuple[np.ndarray, np.ndarray]:
    attrs = json.loads((group_path / ".zattrs").read_text())
    multiscale = attrs["multiscales"][0]
    dataset = next(ds for ds in multiscale["datasets"] if ds["path"] == level_path)

    scale = None
    translation = None
    for transform in dataset["coordinateTransformations"]:
        if transform["type"] == "scale":
            scale = transform["scale"]
        elif transform["type"] == "translation":
            translation = transform["translation"]

    if scale is None:
        scale = [1.0, 1.0, 1.0]
    if translation is None:
        translation = [0.0, 0.0, 0.0]
    return np.asarray(scale, dtype=np.float64), np.asarray(translation, dtype=np.float64)


def _level_transform_from_dataset(dataset: dict) -> tuple[np.ndarray, np.ndarray]:
    scale = None
    translation = None
    for transform in dataset["coordinateTransformations"]:
        if transform["type"] == "scale":
            scale = transform["scale"]
        elif transform["type"] == "translation":
            translation = transform["translation"]
    if scale is None:
        scale = [1.0, 1.0, 1.0]
    if translation is None:
        translation = [0.0, 0.0, 0.0]
    return np.asarray(scale, dtype=np.float64), np.asarray(translation, dtype=np.float64)


def find_matching_gt_level(
    gt_class_path: Path,
    pred_scale: np.ndarray,
    pred_translation: np.ndarray,
) -> str:
    attrs = json.loads((gt_class_path / ".zattrs").read_text())
    datasets = attrs["multiscales"][0]["datasets"]
    for dataset in datasets:
        level_path = dataset["path"]
        scale, _ = _level_transform_from_dataset(dataset)
        if np.allclose(scale, pred_scale):
            return level_path

    for dataset in datasets:
        level_path = dataset["path"]
        scale, translation = _level_transform_from_dataset(dataset)
        if np.allclose(scale[1:], pred_scale[1:]) and np.allclose(
            translation, pred_translation, atol=1e-3
        ):
            return level_path

    best_dataset = min(
        datasets,
        key=lambda ds: float(
            np.linalg.norm(_level_transform_from_dataset(ds)[1] - pred_translation)
        ),
    )
    return best_dataset["path"]


def crop_gt_to_prediction_region(
    gt_class_path: Path,
    gt_level: str,
    pred_shape: tuple[int, int, int],
    pred_scale: np.ndarray,
    pred_translation: np.ndarray,
) -> np.ndarray:
    gt_scale, gt_translation = read_level_transform(gt_class_path, gt_level)
    gt_array = zarr.open(str(gt_class_path / gt_level), mode="r")
    start = np.rint((pred_translation - gt_translation) / gt_scale).astype(int)
    stop = start + np.asarray(pred_shape, dtype=int)

    gt_shape = np.asarray(gt_array.shape, dtype=int)
    clipped_start = np.maximum(start, 0)
    clipped_stop = np.minimum(stop, gt_shape)
    if np.any(clipped_start >= clipped_stop):
        raise ValueError(
            f"Prediction region start={start.tolist()}, stop={stop.tolist()} has no overlap with "
            f"GT shape {gt_array.shape} for {gt_class_path / gt_level}"
        )

    slices = tuple(slice(int(clipped_start[i]), int(clipped_stop[i])) for i in range(3))
    label = (np.asarray(gt_array[slices]) > 0).astype(np.uint8)
    if label.shape != pred_shape:
        label_tensor = torch.from_numpy(label[None, None].astype(np.float32))
        resized = torch.nn.functional.interpolate(
            label_tensor,
            size=pred_shape,
            mode="nearest",
        )
        label = resized[0, 0].numpy().astype(np.uint8)
    return label


def load_logits(crop_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arrays = []
    for cls in CLASSES:
        arr_path = crop_path / cls / "s0"
        if not arr_path.exists():
            raise FileNotFoundError(f"Missing prediction array: {arr_path}")
        arrays.append(np.asarray(zarr.open(str(arr_path), mode="r")))

    shapes = {arr.shape for arr in arrays}
    if len(shapes) != 1:
        raise ValueError(f"Prediction arrays have different shapes: {shapes}")
    pred_scale, pred_translation = read_level_transform(crop_path / CLASSES[0], "s0")
    return np.stack(arrays, axis=0), pred_scale, pred_translation


def load_binary_labels(
    dataset: str,
    crop: str,
    pred_shape: tuple[int, int, int],
    pred_scale: np.ndarray,
    pred_translation: np.ndarray,
) -> np.ndarray:
    crop_path = (
        DATA_DIR
        / dataset
        / f"{dataset}.zarr"
        / "recon-1"
        / "labels"
        / "groundtruth"
        / crop
    )

    arrays = []
    gt_level = find_matching_gt_level(crop_path / CLASSES[0], pred_scale, pred_translation)
    for cls in CLASSES:
        class_path = crop_path / cls
        if not (class_path / gt_level).exists():
            raise FileNotFoundError(f"Missing label array: {class_path / gt_level}")
        arrays.append(
            crop_gt_to_prediction_region(
                class_path,
                gt_level,
                pred_shape,
                pred_scale,
                pred_translation,
            )
        )

    shapes = {arr.shape for arr in arrays}
    if len(shapes) != 1:
        raise ValueError(f"Label arrays have different shapes: {shapes}")
    return np.stack(arrays, axis=0)


def iter_prediction_crops() -> list[tuple[str, str, Path]]:
    crops = []
    for dataset_dir in sorted(PREDICTIONS_DIR.glob("*.zarr")):
        if not dataset_dir.is_dir():
            continue
        dataset = dataset_dir.name.removesuffix(".zarr")
        for crop_dir in sorted(dataset_dir.glob("crop*")):
            if crop_dir.is_dir():
                crops.append((dataset, crop_dir.name, crop_dir))
    return crops


def class_volume_fractions(gt_counts: np.ndarray) -> np.ndarray:
    total = float(gt_counts.sum())
    if total <= 0:
        return np.full_like(gt_counts, np.nan, dtype=np.float64)
    return gt_counts.astype(np.float64) / total


def print_metric_row(name: str, per_class: np.ndarray, gt_fractions: np.ndarray) -> None:
    per_class = np.asarray(per_class, dtype=np.float64)
    mean_value = float(np.nanmean(per_class))
    valid = ~np.isnan(per_class) & ~np.isnan(gt_fractions)
    if valid.any():
        weights = gt_fractions[valid] / gt_fractions[valid].sum()
        total_value = float(np.sum(per_class[valid] * weights))
    else:
        total_value = float("nan")

    values = ", ".join(
        f"{cls}={format_metric(float(score))}"
        for cls, score in zip(CLASSES, per_class)
    )
    print(
        f"{name}: mean={format_metric(mean_value)}, "
        f"total_by_gt_volume={format_metric(total_value)} | {values}"
    )


def metric_scores(
    logits: np.ndarray,
    pred: np.ndarray,
    labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if DiceMetric is None or SurfaceDiceMetric is None:
        raise RuntimeError(
            "MONAI is required for Dice/NS Dice metrics. "
            "Run this script in the math environment where monai is installed."
        )

    pred_one_hot = np.eye(len(CLASSES), dtype=np.float32)[pred].transpose(3, 0, 1, 2)
    gt_one_hot = labels.astype(np.float32)
    prob_softmax = torch.softmax(torch.from_numpy(logits).float(), dim=0).numpy()

    pred_tensor = torch.from_numpy(pred_one_hot).unsqueeze(0)
    gt_tensor = torch.from_numpy(gt_one_hot).unsqueeze(0)
    prob_tensor = torch.from_numpy(prob_softmax).unsqueeze(0)

    dice = DiceMetric(include_background=True, reduction="mean_batch", ignore_empty=False)
    one_hot = dice(y_pred=pred_tensor, y=gt_tensor).detach().cpu().numpy().reshape(-1)
    dice.reset()

    logits_dice = dice(y_pred=prob_tensor, y=gt_tensor).detach().cpu().numpy().reshape(-1)
    dice.reset()

    ns_dice = SurfaceDiceMetric(
        class_thresholds=[NS_DICE_DISTANCE_TOLERANCE] * len(CLASSES),
        include_background=True,
        reduction="mean_batch",
    )
    ns = ns_dice(y_pred=pred_tensor, y=gt_tensor).detach().cpu().numpy().reshape(-1)
    ns_dice.reset()

    return one_hot, logits_dice, ns


def evaluate_crop(dataset: str, crop: str, crop_path: Path) -> CropMetrics:
    logits, pred_scale, pred_translation = load_logits(crop_path)
    labels = load_binary_labels(
        dataset,
        crop,
        logits.shape[1:],
        pred_scale,
        pred_translation,
    )
    if logits.shape[1:] != labels.shape[1:]:
        raise ValueError(
            f"{dataset}/{crop}: prediction shape {logits.shape[1:]} "
            f"does not match label shape {labels.shape[1:]}"
        )

    pred = np.argmax(logits, axis=0)
    label_sum = labels.sum(axis=0)
    valid = label_sum == 1
    gt = np.argmax(labels, axis=0)

    correct = int(((pred == gt) & valid).sum())
    valid_voxels = int(valid.sum())
    gt_counts = labels.reshape(labels.shape[0], -1).sum(axis=1).astype(np.float64)
    one_hot, logits_dice, ns = metric_scores(logits, pred, labels)

    return CropMetrics(
        dataset=dataset,
        crop=crop,
        valid_voxels=valid_voxels,
        correct_voxels=correct,
        gt_counts=gt_counts,
        one_hot_dice=one_hot,
        logits_dice=logits_dice,
        ns_dice=ns,
    )


def weighted_average(metrics: list[CropMetrics], attr: str) -> np.ndarray:
    total_voxels = sum(m.valid_voxels for m in metrics)
    if total_voxels <= 0:
        return np.full(len(CLASSES), np.nan, dtype=np.float64)

    weighted_sum = np.zeros(len(CLASSES), dtype=np.float64)
    weight_sum = np.zeros(len(CLASSES), dtype=np.float64)
    for metric in metrics:
        scores = np.asarray(getattr(metric, attr), dtype=np.float64)
        valid = ~np.isnan(scores)
        weighted_sum[valid] += scores[valid] * metric.valid_voxels
        weight_sum[valid] += metric.valid_voxels

    result = np.full(len(CLASSES), np.nan, dtype=np.float64)
    valid = weight_sum > 0
    result[valid] = weighted_sum[valid] / weight_sum[valid]
    return result


def main() -> None:
    crops = iter_prediction_crops()
    if not crops:
        raise RuntimeError(f"No prediction crops found under {PREDICTIONS_DIR}")

    print(f"Prediction root: {PREDICTIONS_DIR}")
    print(f"Found {len(crops)} prediction crop folders.")
    print()

    metrics: list[CropMetrics] = []
    failures: list[str] = []
    for index, (dataset, crop, crop_path) in enumerate(crops, start=1):
        try:
            metric = evaluate_crop(dataset, crop, crop_path)
            metrics.append(metric)
            accuracy = metric.correct_voxels / max(1, metric.valid_voxels)
            print(
                f"[{index:>3}/{len(crops)}] {dataset}/{crop}: "
                f"valid={metric.valid_voxels:,}, accuracy={format_percent(accuracy)}"
            )
        except Exception as exc:
            failures.append(f"{dataset}/{crop}: {exc}")
            print(f"[{index:>3}/{len(crops)}] SKIP {dataset}/{crop}: {exc}")

    if not metrics:
        raise RuntimeError("No crops could be evaluated.")

    total_valid = sum(m.valid_voxels for m in metrics)
    total_correct = sum(m.correct_voxels for m in metrics)
    overall_accuracy = total_correct / max(1, total_valid)
    gt_counts = np.sum([m.gt_counts for m in metrics], axis=0)
    gt_fractions = class_volume_fractions(gt_counts)

    print()
    print("=" * 80)
    print("Overall metrics")
    print(f"Evaluated crops: {len(metrics)} / {len(crops)}")
    print(
        f"Accuracy: {total_correct:,} / {total_valid:,} valid voxels "
        f"({format_percent(overall_accuracy)})"
    )
    print()
    print("Global GT class volume weights:")
    print(
        ", ".join(
            f"{cls}={format_percent(float(frac))}"
            for cls, frac in zip(CLASSES, gt_fractions)
        )
    )
    print()
    print(
        "Dice rows below are crop-size weighted averages of per-crop scores. "
        "mean = equal class vote; total_by_gt_volume = class scores weighted "
        "by the global GT class volume fractions above."
    )
    print_metric_row("One-hot Dice", weighted_average(metrics, "one_hot_dice"), gt_fractions)
    print_metric_row("Logits Dice", weighted_average(metrics, "logits_dice"), gt_fractions)
    print_metric_row("NS Dice", weighted_average(metrics, "ns_dice"), gt_fractions)

    if failures:
        print()
        print("=" * 80)
        print(f"Skipped crops: {len(failures)}")
        for failure in failures:
            print(f"- {failure}")


if __name__ == "__main__":
    main()
