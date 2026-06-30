from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import zarr
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch


CROP_ID = "crop139"
DATASET = "jrc_mus-liver"

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


def load_binary_labels(
    crop_path: Path, classes: list[str], level_path: str = "s1"
) -> np.ndarray:
    arrays = []
    for cls in classes:
        arr_path = crop_path / cls / level_path
        if not arr_path.exists():
            arr_path = crop_path / cls / "s0"
        if not arr_path.exists():
            raise FileNotFoundError(f"Missing label array: {arr_path}")
        arrays.append((np.asarray(zarr.open(str(arr_path), mode="r")) > 0).astype(np.uint8))

    shapes = {arr.shape for arr in arrays}
    if len(shapes) != 1:
        raise ValueError(f"Label arrays have different shapes: {shapes}")
    return np.stack(arrays, axis=0)


def _read_level_transform(group_path: Path, level_path: str = "s0") -> tuple[list[float], list[float]]:
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
    return scale, translation


def load_raw_crop(pred_crop_path: Path, raw_path: Path, pred_shape: tuple[int, int, int]) -> np.ndarray:
    pred_scale, pred_translation = _read_level_transform(pred_crop_path / CLASSES[0])
    raw_scale, raw_translation = _read_level_transform(raw_path)

    if not np.allclose(pred_scale, raw_scale):
        raise ValueError(f"Prediction scale {pred_scale} does not match raw scale {raw_scale}")

    start = [
        int(round((pred_translation[i] - raw_translation[i]) / raw_scale[i]))
        for i in range(3)
    ]
    stop = [start[i] + pred_shape[i] for i in range(3)]

    raw = zarr.open(str(raw_path / "s0"), mode="r")
    slices = tuple(slice(start[i], stop[i]) for i in range(3))
    crop = np.asarray(raw[slices])
    if crop.shape != pred_shape:
        raise ValueError(f"Raw crop shape {crop.shape} does not match prediction shape {pred_shape}")
    return crop


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
    labels = load_binary_labels(GT_CROP, CLASSES, level_path="s1")
    gt = print_groundtruth_stats(labels, CLASSES)
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
