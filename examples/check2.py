from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import zarr


CLASSES = ["endo_lum", "cyto", "endo_mem", "pm", "ecs"]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRED_CROP = (
    PROJECT_ROOT
    / "data"
    / "predictions"
    / "jrc_mus-liver.zarr"
    / "crop124"
)
GT_CROP = (
    PROJECT_ROOT
    / "data"
    / "jrc_mus-liver"
    / "jrc_mus-liver.zarr"
    / "recon-1"
    / "labels"
    / "groundtruth"
    / "crop124"
)


def load_array(group_path: Path, preferred_level: str) -> np.ndarray:
    candidates = [preferred_level, "s0", "s1", "s2", "s3", "s4"]
    for level in dict.fromkeys(candidates):
        array_path = group_path / level
        if array_path.exists():
            return np.asarray(zarr.open(str(array_path), mode="r"))
    raise FileNotFoundError(f"No scale array found in {group_path}")


def load_logits() -> np.ndarray:
    arrays = [
        load_array(PRED_CROP / class_name, "s0").astype(np.float32)
        for class_name in CLASSES
    ]
    shapes = {array.shape for array in arrays}
    if len(shapes) != 1:
        raise ValueError(f"Prediction shapes do not match: {shapes}")
    return np.stack(arrays)


def load_groundtruth(expected_shape: tuple[int, ...]) -> np.ndarray:
    arrays = []
    for class_name in CLASSES:
        class_path = GT_CROP / class_name
        matching_array = None
        for level in ("s0", "s1", "s2", "s3", "s4"):
            array_path = class_path / level
            if not array_path.exists():
                continue
            candidate = np.asarray(zarr.open(str(array_path), mode="r"))
            if candidate.shape == expected_shape:
                matching_array = candidate
                break
        if matching_array is None:
            raise ValueError(
                f"No ground-truth scale for {class_name} matches {expected_shape}"
            )
        arrays.append((matching_array > 0).astype(np.uint8))
    return np.stack(arrays)


def add_pixel_ticks(axis, image: np.ndarray, step: int = 25) -> None:
    axis.set_xticks(np.arange(0, image.shape[1], step))
    axis.set_yticks(np.arange(0, image.shape[0], step))
    axis.tick_params(axis="both", labelsize=8)


def print_statistics(logits: np.ndarray, groundtruth: np.ndarray) -> None:
    voxel_count = int(np.prod(logits.shape[1:]))
    probabilities = 1.0 / (1.0 + np.exp(-logits))

    print(f"Prediction path: {PRED_CROP}")
    print(f"Groundtruth path: {GT_CROP}")
    print(f"Shape: {logits.shape} = [classes, z, y, x]")
    print()

    for index, class_name in enumerate(CLASSES):
        gt_count = int(groundtruth[index].sum())
        predicted_count = int((probabilities[index] >= 0.5).sum())
        print(
            f"{class_name:>8}: "
            f"GT={gt_count:,} ({100.0 * gt_count / voxel_count:.4f}%), "
            f"sigmoid>=0.5={predicted_count:,} "
            f"({100.0 * predicted_count / voxel_count:.4f}%), "
            f"logit[min={logits[index].min():.4f}, "
            f"max={logits[index].max():.4f}, "
            f"mean={logits[index].mean():.4f}]"
        )


def plot_groundtruth(groundtruth: np.ndarray, output_path: Path) -> None:
    z_index = groundtruth.shape[1] // 2
    figure, axes = plt.subplots(
        1,
        len(CLASSES),
        figsize=(4.5 * len(CLASSES), 4),
        constrained_layout=True,
    )

    for index, (axis, class_name) in enumerate(zip(axes, CLASSES)):
        image = groundtruth[index, z_index]
        axis.imshow(image, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        axis.set_title(f"{class_name} GT z={z_index}")
        add_pixel_ticks(axis, image)

    figure.suptitle("Five-class ground-truth masks")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_logits(logits: np.ndarray, output_path: Path) -> None:
    z_index = logits.shape[1] // 2
    figure, axes = plt.subplots(
        1,
        len(CLASSES),
        figsize=(4.5 * len(CLASSES), 4),
        constrained_layout=True,
    )

    for index, (axis, class_name) in enumerate(zip(axes, CLASSES)):
        image = logits[index, z_index]
        plot = axis.imshow(image, cmap="coolwarm")
        axis.set_title(f"{class_name} logit z={z_index}")
        add_pixel_ticks(axis, image)
        figure.colorbar(plot, ax=axis, fraction=0.046, pad=0.04)

    figure.suptitle("Five-class raw logits before sigmoid")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    logits = load_logits()
    groundtruth = load_groundtruth(logits.shape[1:])
    print_statistics(logits, groundtruth)

    groundtruth_output = Path(__file__).with_name(
        "check2_crop124_groundtruth.png"
    )
    logits_output = Path(__file__).with_name("check2_crop124_logits.png")

    plot_groundtruth(groundtruth, groundtruth_output)
    plot_logits(logits, logits_output)

    print()
    print(f"Wrote: {groundtruth_output}")
    print(f"Wrote: {logits_output}")


if __name__ == "__main__":
    main()
