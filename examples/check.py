from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import torch
import zarr
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

try:
    from monai.metrics import DiceMetric, SurfaceDiceMetric
except ImportError:
    DiceMetric = None
    SurfaceDiceMetric = None


CROP_ID = "crop164"
DATASET = "jrc_mus-kidney"

PRED_CROP = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "predictions"
    / f"{DATASET}.zarr"
    / CROP_ID
)
GT_CROP = (
    Path(__file__).resolve().parents[1]
    / "data"
    / DATASET
    / f"{DATASET}.zarr"
    / "recon-1"
    / "labels"
    / "groundtruth"
    / CROP_ID
)
RAW_DATA = (
    Path(__file__).resolve().parents[1]
    / "data"
    / DATASET
    / f"{DATASET}.zarr"
    / "recon-1"
    / "em"
    / "fibsem-uint8"
)
CLASSES = ["endo_lum", "cyto", "endo_mem", "pm", "ecs", 'bg']
COLORS = ["#39a9db", "#50c878", "#f06a6a", "#f2c14e", "#2a9d8f", "#5e4fa2"]
NS_DICE_DISTANCE_TOLERANCE = 1.0
CELLMAP_DICE_SMOOTH = 0.02
CELLMAP_DICE_INCLUDE_BACKGROUND = True


def set_pixel_ticks(ax, image: np.ndarray, step: int = 25) -> None:
    y_size, x_size = image.shape[:2]
    ax.set_xticks(np.arange(0, x_size, step))
    ax.set_yticks(np.arange(0, y_size, step))
    ax.tick_params(axis="both", labelsize=8)


def load_logits(crop_path: Path, classes: list[str]) -> np.ndarray:
    arrays = []
    for cls in classes:
        arr_path = crop_path / cls / "s0"
        if not arr_path.exists():
            raise FileNotFoundError(f"Missing prediction array: {arr_path}")
        arrays.append(np.asarray(zarr.open(str(arr_path), mode="r")))

    shapes = {arr.shape for arr in arrays}
    if len(shapes) != 1:
        raise ValueError(f"Prediction arrays have different shapes: {shapes}")
    return np.stack(arrays, axis=0)


def _read_level_transform(group_path: Path, level_path: str = "s0") -> tuple[np.ndarray, np.ndarray]:
    attrs_path = group_path / ".zattrs"
    attrs = json.loads(attrs_path.read_text())
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
    gt_scale, gt_translation = _read_level_transform(gt_class_path, gt_level)
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


def load_binary_labels(
    crop_path: Path,
    classes: list[str],
    pred_shape: tuple[int, int, int],
    pred_scale: np.ndarray,
    pred_translation: np.ndarray,
) -> np.ndarray:
    arrays = []
    gt_level = find_matching_gt_level(crop_path / classes[0], pred_scale, pred_translation)
    print(
        f"GT alignment: using level {gt_level}, prediction scale={pred_scale.tolist()}, "
        f"translation={pred_translation.tolist()}"
    )
    for cls in classes:
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


def crop_raw_to_prediction_region(
    raw_path: Path,
    raw_level: str,
    pred_shape: tuple[int, int, int],
    pred_translation: np.ndarray,
) -> np.ndarray:
    raw_scale, raw_translation = _read_level_transform(raw_path, raw_level)
    raw_array = zarr.open(str(raw_path / raw_level), mode="r")
    start = np.rint((pred_translation - raw_translation) / raw_scale).astype(int)
    stop = start + np.asarray(pred_shape, dtype=int)

    raw_shape = np.asarray(raw_array.shape, dtype=int)
    clipped_start = np.maximum(start, 0)
    clipped_stop = np.minimum(stop, raw_shape)
    if np.any(clipped_start >= clipped_stop):
        raise ValueError(
            f"Prediction region start={start.tolist()}, stop={stop.tolist()} has no overlap with "
            f"raw shape {raw_array.shape} for {raw_path / raw_level}"
        )

    slices = tuple(slice(int(clipped_start[i]), int(clipped_stop[i])) for i in range(3))
    crop = np.asarray(raw_array[slices])
    if crop.shape != pred_shape:
        crop_tensor = torch.from_numpy(crop[None, None].astype(np.float32))
        resized = torch.nn.functional.interpolate(
            crop_tensor,
            size=pred_shape,
            mode="trilinear",
            align_corners=False,
        )
        crop = resized[0, 0].numpy()
    return crop


def load_raw_crop(pred_crop_path: Path, raw_path: Path, pred_shape: tuple[int, int, int]) -> np.ndarray:
    pred_scale, pred_translation = _read_level_transform(pred_crop_path / CLASSES[0])
    raw_level = find_matching_gt_level(raw_path, pred_scale, pred_translation)
    print(f"Raw alignment: using level {raw_level}")
    return crop_raw_to_prediction_region(raw_path, raw_level, pred_shape, pred_translation)


def print_stats(logits: np.ndarray, classes: list[str]) -> np.ndarray:
    all_zero = np.all(logits == 0, axis=0)
    valid = ~all_zero
    pred = np.argmax(logits, axis=0)

    print(f"Crop path: {PRED_CROP}")
    print(f"Logits shape: {logits.shape} = [classes, z, y, x]")
    print(f"All-zero voxels: {all_zero.sum():,} / {all_zero.size:,}")
    print(f"NaN voxels: {np.isnan(logits).any(axis=0).sum():,} / {all_zero.size:,}")
    print()

    for i, cls in enumerate(classes):
        arr = logits[i]
        count = int(((pred == i) & valid).sum())
        percent = 100.0 * count / max(1, int(valid.sum()))
        positive = int((arr >= 0).sum())
        positive_percent = 100.0 * positive / arr.size
        print(
            f"{cls:>8}: min={arr.min(): .4f}, max={arr.max(): .4f}, "
            f"mean={arr.mean(): .4f}, sigmoid>=0.5={positive:,} "
            f"({positive_percent:.2f}%), dominant={count:,} ({percent:.2f}%)"
        )
    return pred


def print_groundtruth_stats(labels: np.ndarray, classes: list[str]) -> np.ndarray:
    label_sum = labels.sum(axis=0)
    overlap = label_sum > 1
    unlabeled = label_sum == 0
    gt = np.argmax(labels, axis=0)

    print()
    print(f"Groundtruth path: {GT_CROP}")
    print(f"Groundtruth shape: {labels.shape} = [classes, z, y, x]")
    print(
        f"Unlabeled voxels among these {len(classes)} classes: "
        f"{unlabeled.sum():,} / {unlabeled.size:,}"
    )
    print(
        f"Overlapping voxels among these {len(classes)} classes: "
        f"{overlap.sum():,} / {overlap.size:,}"
    )
    print()

    total = labels.shape[1] * labels.shape[2] * labels.shape[3]
    for i, cls in enumerate(classes):
        count = int(labels[i].sum())
        percent = 100.0 * count / max(1, total)
        print(f"GT {cls:>8}: {count:,} ({percent:.4f}%)")
    return gt


def print_accuracy(pred: np.ndarray, labels: np.ndarray) -> None:
    label_sum = labels.sum(axis=0)
    valid = label_sum == 1
    gt = np.argmax(labels, axis=0)

    if pred.shape != gt.shape:
        raise ValueError(f"Prediction shape {pred.shape} does not match GT shape {gt.shape}")

    correct = (pred == gt) & valid
    accuracy = 100.0 * correct.sum() / max(1, valid.sum())

    print()
    print(
        f"Accuracy: {correct.sum():,} / {valid.sum():,} valid voxels "
        f"({accuracy:.4f}%)"
    )


def _format_metric(value: float) -> str:
    if np.isnan(value):
        return "nan"
    return f"{value:.4f}"


def class_volume_fractions(labels: np.ndarray) -> np.ndarray:
    class_counts = labels.reshape(labels.shape[0], -1).sum(axis=1).astype(np.float64)
    total = float(class_counts.sum())
    if total <= 0:
        return np.full(labels.shape[0], np.nan, dtype=np.float64)
    return class_counts / total


def _print_metric_row(
    name: str,
    per_class: np.ndarray,
    class_fractions: np.ndarray | None = None,
) -> None:
    per_class = np.asarray(per_class, dtype=np.float64).reshape(-1)
    mean_value = float(np.nanmean(per_class))
    total_text = ""
    if class_fractions is not None:
        class_fractions = np.asarray(class_fractions, dtype=np.float64).reshape(-1)
        valid = ~np.isnan(per_class) & ~np.isnan(class_fractions)
        if valid.any():
            normalized_fractions = class_fractions[valid] / class_fractions[valid].sum()
            total_value = float(np.sum(per_class[valid] * normalized_fractions))
            total_text = f", total_by_gt_volume={_format_metric(total_value)}"

    values = ", ".join(
        f"{cls}={_format_metric(float(score))}"
        for cls, score in zip(CLASSES, per_class)
    )
    print(f"{name}: mean={_format_metric(mean_value)}{total_text} | {values}")


def print_class_volume_weights(labels: np.ndarray) -> np.ndarray:
    fractions = class_volume_fractions(labels)
    values = ", ".join(
        f"{cls}={fraction * 100:.4f}%"
        for cls, fraction in zip(CLASSES, fractions)
    )
    print()
    print("GT class volume weights for total/global Dice")
    print(f"These weights are used only for the check.py total_by_gt_volume metric: {values}")
    return fractions


def cellmap_loss_style_dice(logits: np.ndarray, labels: np.ndarray) -> np.ndarray:
    outputs = torch.from_numpy(logits).float().unsqueeze(0)
    targets = torch.from_numpy(labels).float().unsqueeze(0)
    target_indices = targets.argmax(dim=1)
    target_one_hot = torch.nn.functional.one_hot(
        target_indices,
        num_classes=outputs.shape[1],
    ).movedim(-1, 1)
    target_one_hot = target_one_hot.to(dtype=outputs.dtype)
    probabilities = torch.softmax(outputs, dim=1)

    if not CELLMAP_DICE_INCLUDE_BACKGROUND and outputs.shape[1] > 1:
        probabilities = probabilities[:, :-1]
        target_one_hot = target_one_hot[:, :-1]

    reduce_dims = tuple(range(2, outputs.ndim))
    intersection = (probabilities * target_one_hot).sum(dim=reduce_dims)
    denominator = probabilities.sum(dim=reduce_dims) + target_one_hot.sum(dim=reduce_dims)
    dice_score = (2 * intersection + CELLMAP_DICE_SMOOTH) / (
        denominator + CELLMAP_DICE_SMOOTH
    )
    return dice_score.squeeze(0).detach().cpu().numpy()


def print_monai_dice_metrics(logits: np.ndarray, pred: np.ndarray, labels: np.ndarray) -> None:
    if DiceMetric is None or SurfaceDiceMetric is None:
        print()
        print("MONAI Dice metrics skipped: please install monai in this Python environment.")
        return

    pred_one_hot = np.eye(len(CLASSES), dtype=np.float32)[pred].transpose(3, 0, 1, 2)
    gt_one_hot = labels.astype(np.float32)
    prob_softmax = torch.softmax(torch.from_numpy(logits).float(), dim=0).numpy()
    class_fractions = print_class_volume_weights(labels)

    pred_tensor = torch.from_numpy(pred_one_hot).unsqueeze(0)
    gt_tensor = torch.from_numpy(gt_one_hot).unsqueeze(0)
    prob_tensor = torch.from_numpy(prob_softmax).unsqueeze(0)

    dice = DiceMetric(include_background=True, reduction="mean_batch", ignore_empty=False)

    one_hot_scores = dice(y_pred=pred_tensor, y=gt_tensor).detach().cpu().numpy()
    dice.reset()

    logits_scores = dice(y_pred=prob_tensor, y=gt_tensor).detach().cpu().numpy()
    dice.reset()

    ns_dice = SurfaceDiceMetric(
        class_thresholds=[NS_DICE_DISTANCE_TOLERANCE] * len(CLASSES),
        include_background=True,
        reduction="mean_batch",
    )
    ns_scores = ns_dice(y_pred=pred_tensor, y=gt_tensor).detach().cpu().numpy()
    ns_dice.reset()

    print()
    print("MONAI segmentation metrics")
    print(
        "MONAI Logits Dice uses softmax probability maps. The separate "
        "CellMap loss-style Dice below directly reproduces the Dice formula "
        "used by CellMapDiceCELoss / CellMapFilteredDynamicWeightedDiceCELoss."
    )
    print(f"NS Dice tolerance: {NS_DICE_DISTANCE_TOLERANCE} voxel")
    _print_metric_row("One-hot Dice", one_hot_scores, class_fractions)
    _print_metric_row("Logits Dice", logits_scores, class_fractions)
    cellmap_scores = cellmap_loss_style_dice(logits, labels)
    print(
        f"CellMap loss-style Dice smooth={CELLMAP_DICE_SMOOTH}, "
        f"include_background={CELLMAP_DICE_INCLUDE_BACKGROUND}"
    )
    _print_metric_row("CellMap loss-style Dice", cellmap_scores, class_fractions)
    _print_metric_row("NS Dice", ns_scores, class_fractions)


def plot_argmax_slices(
    pred: np.ndarray, out_path: Path, figure_title: str = "Prediction argmax slices"
) -> None:
    z_mid = pred.shape[0] // 2
    y_mid = pred.shape[1] // 2
    x_mid = pred.shape[2] // 2

    cmap = ListedColormap(COLORS)
    slices = [
        (pred[z_mid, :, :], f"argmax z={z_mid}", "YX"),
        (pred[:, y_mid, :], f"argmax y={y_mid}", "ZX"),
        (pred[:, :, x_mid], f"argmax x={x_mid}", "ZY"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for ax, (image, title, axis_label) in zip(axes, slices):
        ax.imshow(image, cmap=cmap, vmin=0, vmax=len(CLASSES) - 1, interpolation="nearest")
        ax.set_title(f"{title} ({axis_label})")
        set_pixel_ticks(ax, image)

    handles = [Patch(color=color, label=cls) for cls, color in zip(CLASSES, COLORS)]
    fig.legend(handles=handles, loc="lower center", ncol=len(CLASSES))
    fig.suptitle(figure_title)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_logit_slices(logits: np.ndarray, out_path: Path) -> None:
    z_mid = logits.shape[1] // 2

    fig, axes = plt.subplots(
        1,
        len(CLASSES),
        figsize=(4.5 * len(CLASSES), 4),
        constrained_layout=True,
    )
    for i, (ax, cls) in enumerate(zip(axes, CLASSES)):
        image = logits[i, z_mid, :, :]
        im = ax.imshow(image, cmap="coolwarm")
        ax.set_title(f"{cls} logit z={z_mid}")
        set_pixel_ticks(ax, image)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Raw logits before argmax")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_raw_overlay(raw_crop: np.ndarray, pred: np.ndarray, out_path: Path) -> None:
    z_mid = raw_crop.shape[0] // 2
    raw_slice = raw_crop[z_mid]
    pred_slice = pred[z_mid]

    cmap = ListedColormap(COLORS)
    overlay_alpha = np.full(pred_slice.shape, 0.38, dtype=np.float32)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    axes[0].imshow(raw_slice, cmap="gray")
    axes[0].set_title(f"raw EM z={z_mid}")

    axes[1].imshow(pred_slice, cmap=cmap, vmin=0, vmax=len(CLASSES) - 1, interpolation="nearest")
    axes[1].set_title("prediction argmax")

    axes[2].imshow(raw_slice, cmap="gray")
    axes[2].imshow(
        pred_slice,
        cmap=cmap,
        vmin=0,
        vmax=len(CLASSES) - 1,
        alpha=overlay_alpha,
        interpolation="nearest",
    )
    axes[2].set_title("raw + prediction overlay")

    for ax, image in zip(axes, [raw_slice, pred_slice, raw_slice]):
        set_pixel_ticks(ax, image)

    handles = [Patch(color=color, label=cls) for cls, color in zip(CLASSES, COLORS)]
    fig.legend(handles=handles, loc="lower center", ncol=len(CLASSES))
    fig.suptitle("Prediction overlay on raw EM")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_groundtruth_overlay(raw_crop: np.ndarray, gt: np.ndarray, out_path: Path) -> None:
    z_mid = raw_crop.shape[0] // 2
    raw_slice = raw_crop[z_mid]
    gt_slice = gt[z_mid]

    cmap = ListedColormap(COLORS)
    overlay_alpha = np.full(gt_slice.shape, 0.38, dtype=np.float32)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    axes[0].imshow(raw_slice, cmap="gray")
    axes[0].set_title(f"raw EM z={z_mid}")

    axes[1].imshow(gt_slice, cmap=cmap, vmin=0, vmax=len(CLASSES) - 1, interpolation="nearest")
    axes[1].set_title("groundtruth argmax")

    axes[2].imshow(raw_slice, cmap="gray")
    axes[2].imshow(
        gt_slice,
        cmap=cmap,
        vmin=0,
        vmax=len(CLASSES) - 1,
        alpha=overlay_alpha,
        interpolation="nearest",
    )
    axes[2].set_title("raw + groundtruth overlay")

    for ax, image in zip(axes, [raw_slice, gt_slice, raw_slice]):
        set_pixel_ticks(ax, image)

    handles = [Patch(color=color, label=cls) for cls, color in zip(CLASSES, COLORS)]
    fig.legend(handles=handles, loc="lower center", ncol=len(CLASSES))
    fig.suptitle("Groundtruth overlay on raw EM")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    logits = load_logits(PRED_CROP, CLASSES)
    pred = print_stats(logits, CLASSES)
    pred_scale, pred_translation = _read_level_transform(PRED_CROP / CLASSES[0], "s0")
    labels = load_binary_labels(
        GT_CROP,
        CLASSES,
        logits.shape[1:],
        pred_scale,
        pred_translation,
    )
    gt = print_groundtruth_stats(labels, CLASSES)
    print_accuracy(pred, labels)
    print_monai_dice_metrics(logits, pred, labels)
    raw_crop = load_raw_crop(PRED_CROP, RAW_DATA, logits.shape[1:])

    out_dir = Path(__file__).resolve().parent
    output_prefix = f"check_{CROP_ID}"
    argmax_path = out_dir / f"{output_prefix}_argmax.png"
    logits_path = out_dir / f"{output_prefix}_logits.png"
    overlay_path = out_dir / f"{output_prefix}_overlay.png"
    gt_path = out_dir / f"{output_prefix}_groundtruth.png"
    gt_overlay_path = out_dir / f"{output_prefix}_gt_overlay.png"
    plot_argmax_slices(pred, argmax_path, figure_title="Prediction argmax slices")
    plot_logit_slices(logits, logits_path)
    plot_raw_overlay(raw_crop, pred, overlay_path)
    plot_argmax_slices(gt, gt_path, figure_title="Groundtruth argmax slices")
    plot_groundtruth_overlay(raw_crop, gt, gt_overlay_path)

    print()
    print(f"Wrote: {argmax_path}")
    print(f"Wrote: {logits_path}")
    print(f"Wrote: {overlay_path}")
    print(f"Wrote: {gt_path}")
    print(f"Wrote: {gt_overlay_path}")


if __name__ == "__main__":
    main()
