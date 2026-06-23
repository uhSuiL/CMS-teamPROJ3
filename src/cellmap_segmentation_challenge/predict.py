import copy
import itertools
import os
import tempfile
from glob import glob
from typing import Any

import numpy as np
import torch
import torchvision.transforms.v2 as T
from cellmap_data import CellMapDatasetWriter, CellMapImage
from cellmap_data.utils import (
    array_has_singleton_dim,
    is_array_2D,
    permute_singleton_dimension,
)
from cellmap_data.transforms.augment import NaNtoNum
from tqdm import tqdm
from upath import UPath

from .config import CROP_NAME, PREDICTIONS_PATH, RAW_NAME, SEARCH_PATH
from .models import get_model
from .utils import (
    load_safe_config,
    get_test_crops,
    get_test_crop_labels,
    get_data_from_batch,
    get_singleton_dim,
    squeeze_singleton_dim,
    structure_model_output,
    unsqueeze_singleton_dim,
)
from .utils.datasplit import get_formatted_fields, get_raw_path


def _merge_bounds(
    current: dict[str, tuple[float, float]] | None,
    new_bounds: dict[str, tuple[float, float]],
) -> dict[str, tuple[float, float]]:
    if current is None:
        return {axis: (float(lo), float(hi)) for axis, (lo, hi) in new_bounds.items()}
    merged = dict(current)
    for axis, (lo, hi) in new_bounds.items():
        if axis in merged:
            old_lo, old_hi = merged[axis]
            merged[axis] = (min(old_lo, float(lo)), max(old_hi, float(hi)))
        else:
            merged[axis] = (float(lo), float(hi))
    return merged


def _get_crop_target_bounds(
    crop_path: str,
    classes: list[str],
    target_arrays: dict[str, dict],
) -> dict[str, dict[str, tuple[float, float]]]:
    """Infer crop bounds from all available label arrays, not only classes[0]."""
    target_bounds = {}
    for array_name, array_info in target_arrays.items():
        bounds = None
        for label in classes:
            label_path = UPath(crop_path) / label
            try:
                image = CellMapImage(
                    str(label_path),
                    target_class=label,
                    target_scale=array_info["scale"],
                    target_voxel_shape=array_info["shape"],
                    pad=True,
                    pad_value=0,
                )
                bounds = _merge_bounds(bounds, image.bounding_box)
            except Exception:
                continue
        if bounds is None:
            raise ValueError(
                f"Could not infer target bounds for {crop_path}. None of the "
                f"requested classes were readable: {classes}."
            )
        target_bounds[array_name] = bounds
    return target_bounds


def _get_dense_writer_indices(dataset_writer: CellMapDatasetWriter) -> list[int]:
    """Tile prediction centers so the full output extent is covered.

    CellMapDatasetWriter's default writer_indices use roughly one tile per
    patch-sized span of the shrunken sampling box. For crops like 200^3 with
    128^3 patches this can produce a single write tile, leaving the margins
    unwritten. This helper uses enough edge-aligned centers to cover the whole
    output array while still writing normal patch-sized predictions.
    """
    sb = dataset_writer.sampling_box
    bb = dataset_writer.bounding_box
    if sb is None or bb is None:
        return []

    axes = list(sb.keys())
    scale = dataset_writer._write_scale
    patch_shape = dataset_writer._write_voxel_shape
    grid_shape = {
        axis: max(1, int(round((sb[axis][1] - sb[axis][0]) / scale[axis])))
        for axis in axes
    }

    per_axis_positions = []
    for axis in axes:
        full_voxels = max(1, int(round((bb[axis][1] - bb[axis][0]) / scale[axis])))
        patch_voxels = max(1, int(patch_shape[axis]))
        center_grid = grid_shape[axis]
        if full_voxels <= patch_voxels or center_grid <= 1:
            per_axis_positions.append([0])
            continue

        max_start = full_voxels - patch_voxels
        n_tiles = int(np.ceil(full_voxels / patch_voxels))
        starts = sorted(
            {int(round(start)) for start in np.linspace(0, max_start, n_tiles)}
        )
        positions = sorted(
            {
                int(round(start * (center_grid - 1) / max_start))
                for start in starts
            }
        )
        per_axis_positions.append(positions)

    shape_tuple = tuple(grid_shape[axis] for axis in axes)
    indices = []
    for coords in itertools.product(*per_axis_positions):
        flat = 0
        for coord, dim in zip(coords, shape_tuple):
            flat = flat * dim + coord
        indices.append(flat)
    return indices


def _writer_patch_slices(writer, center: dict[str, float], data: np.ndarray):
    """Return destination/source slices for writing one patch."""
    arr_shape = [writer.shape[c] for c in writer.spatial_axes]

    dst_slices = []
    src_slices = []
    for i, axis in enumerate(writer.spatial_axes):
        start_nm = center[axis] - writer.write_world_shape[axis] / 2.0
        start_vox = int(round((start_nm - writer.offset[axis]) / writer.scale[axis]))
        end_vox = start_vox + writer.write_voxel_shape[axis]
        clamp_start = max(0, start_vox)
        clamp_end = min(arr_shape[i], end_vox)
        dst_slices.append(slice(clamp_start, clamp_end))
        src_start = clamp_start - start_vox
        src_slices.append(slice(src_start, src_start + clamp_end - clamp_start))

    while data.ndim > len(writer.spatial_axes) and data.shape[0] == 1:
        data = np.squeeze(data, axis=0)

    actual = tuple(s.stop - s.start for s in dst_slices)
    if data.shape != actual:
        data = data[tuple(src_slices)]

    return tuple(dst_slices), tuple(src_slices), data


def _blend_weight(shape: tuple[int, ...]) -> np.ndarray:
    """Use equal weights so only overlapping voxels are averaged."""
    return np.ones(shape, dtype=np.float32)


def _select_prediction_batch_item(
    outputs: dict[str, Any],
    batch_index: int,
) -> dict[str, Any]:
    """Select one prediction item while preserving the writer's dict structure."""
    item = {}
    for array_name, class_outputs in outputs.items():
        if isinstance(class_outputs, dict):
            item[array_name] = {
                class_name: tensor[batch_index]
                for class_name, tensor in class_outputs.items()
            }
        else:
            item[array_name] = class_outputs[batch_index]
    return item


def _get_owned_prediction_tiles(
    dataset_writer: CellMapDatasetWriter,
) -> list[dict[str, Any]]:
    """Split the output into disjoint regions owned by centered input patches."""
    first_array_writers = next(iter(dataset_writer.target_array_writers.values()))
    first_writer = next(iter(first_array_writers.values()))
    axes = list(first_writer.spatial_axes)
    output_shape = tuple(int(first_writer.shape[axis]) for axis in axes)
    patch_shape = tuple(int(first_writer.write_voxel_shape[axis]) for axis in axes)

    per_axis_regions = []
    for full_size, patch_size in zip(output_shape, patch_shape):
        n_tiles = max(1, int(np.ceil(full_size / patch_size)))
        boundaries = np.rint(np.linspace(0, full_size, n_tiles + 1)).astype(int)
        per_axis_regions.append(
            [(int(boundaries[i]), int(boundaries[i + 1])) for i in range(n_tiles)]
        )

    tiles = []
    for regions in itertools.product(*per_axis_regions):
        center = {}
        dst_slices = []
        src_slices = []
        for axis_index, (axis, region) in enumerate(zip(axes, regions)):
            region_start, region_stop = region
            region_center = (region_start + region_stop) / 2.0
            center[axis] = (
                float(first_writer.offset[axis])
                + region_center * float(first_writer.scale[axis])
            )

            patch_start = int(
                np.floor(region_center - patch_shape[axis_index] / 2.0)
            )
            src_start = region_start - patch_start
            src_stop = src_start + region_stop - region_start
            dst_slices.append(slice(region_start, region_stop))
            src_slices.append(slice(src_start, src_stop))

        tiles.append(
            {
                "center": center,
                "dst_slices": tuple(dst_slices),
                "src_slices": tuple(src_slices),
            }
        )
    return tiles


def _load_prediction_tile_batch(
    dataset_writer: CellMapDatasetWriter,
    tiles: list[dict[str, Any]],
) -> dict[str, Any]:
    """Load model inputs at the centers of a batch of owned output tiles."""
    batch = {}
    for array_name, source in dataset_writer.input_sources.items():
        patches = []
        for tile in tiles:
            patch = source[tile["center"]]
            if patch.ndim > 0 and patch.shape[0] != 1:
                patch = patch.unsqueeze(0)
            patches.append(patch)
        batch[array_name] = torch.stack(patches)
    return batch


def _write_owned_prediction_tile(
    dataset_writer: CellMapDatasetWriter,
    tile: dict[str, Any],
    outputs: dict[str, Any],
) -> None:
    """Write only the disjoint responsibility region of one prediction patch."""
    for array_name, class_outputs in outputs.items():
        if array_name not in dataset_writer.target_array_writers:
            continue
        writers = dataset_writer.target_array_writers[array_name]

        if isinstance(class_outputs, dict):
            items = class_outputs.items()
        else:
            items = (
                (cls, class_outputs[i : i + 1])
                for i, cls in enumerate(dataset_writer.model_classes)
                if class_outputs.ndim > 0 and class_outputs.shape[0] > i
            )

        for cls, tensor in items:
            if cls not in writers:
                continue
            writer = writers[cls]
            data = (
                tensor.detach().cpu().numpy()
                if torch.is_tensor(tensor)
                else np.asarray(tensor)
            )
            while data.ndim > len(writer.spatial_axes) and data.shape[0] == 1:
                data = np.squeeze(data, axis=0)
            writer._zarr_array[tile["dst_slices"]] = data[tile["src_slices"]].astype(
                writer._zarr_array.dtype
            )


def _accumulate_prediction_patch(
    dataset_writer: CellMapDatasetWriter,
    idx: int,
    outputs: dict[str, Any],
    weight_sums: dict[str, np.ndarray],
) -> None:
    """Accumulate overlapping raw logits for later averaging."""
    center = dataset_writer.get_center(int(idx))

    for array_name, class_outputs in outputs.items():
        if array_name not in dataset_writer.target_array_writers:
            continue
        writers = dataset_writer.target_array_writers[array_name]

        if isinstance(class_outputs, dict):
            items = class_outputs.items()
        else:
            items = (
                (cls, class_outputs[i : i + 1])
                for i, cls in enumerate(dataset_writer.model_classes)
                if class_outputs.ndim > 0 and class_outputs.shape[0] > i
            )

        count_updated = False
        for cls, tensor in items:
            if cls not in writers:
                continue
            writer = writers[cls]
            data = tensor.detach().cpu().numpy() if torch.is_tensor(tensor) else np.asarray(tensor)
            dst_slices, src_slices, data = _writer_patch_slices(writer, center, data)
            full_patch_shape = tuple(
                int(writer.write_voxel_shape[axis]) for axis in writer.spatial_axes
            )
            blend_weight = _blend_weight(full_patch_shape)[src_slices]
            arr = writer._zarr_array
            arr[dst_slices] = np.asarray(arr[dst_slices]) + (
                data.astype(np.float32) * blend_weight
            ).astype(arr.dtype)

            if not count_updated:
                weight_sums[array_name][dst_slices] += blend_weight
                count_updated = True


def _average_accumulated_predictions(
    dataset_writer: CellMapDatasetWriter,
    weight_sums: dict[str, np.ndarray],
) -> None:
    """Divide accumulated logits by per-voxel blending weights."""
    for array_name, class_writers in dataset_writer.target_array_writers.items():
        weight_sum = weight_sums[array_name]
        covered = weight_sum > 0
        if not np.any(covered):
            continue
        for writer in class_writers.values():
            arr = writer._zarr_array
            data = np.asarray(arr[:])
            data[covered] = data[covered] / weight_sum[covered]
            arr[:] = data.astype(arr.dtype)


def predict_orthoplanes(
    model: torch.nn.Module, dataset_writer_kwargs: dict[str, Any], batch_size: int
):
    print("Predicting orthogonal planes.")

    # Make a temporary prediction for each axis
    tmp_dir = tempfile.TemporaryDirectory()
    print(f"Temporary directory for predictions: {tmp_dir.name}")
    for axis in range(3):
        # Actually slice per axis by permuting singleton dimension
        temp_kwargs = dataset_writer_kwargs.copy()
        temp_kwargs["target_path"] = os.path.join(
            tmp_dir.name, "output.zarr", str(axis)
        )
        # Permute input_arrays and target_arrays so singleton is at the current axis
        input_arrays = {k: v.copy() for k, v in temp_kwargs["input_arrays"].items()}
        target_arrays = {k: v.copy() for k, v in temp_kwargs["target_arrays"].items()}
        permute_singleton_dimension(input_arrays, axis)
        permute_singleton_dimension(target_arrays, axis)
        temp_kwargs["input_arrays"] = input_arrays
        temp_kwargs["target_arrays"] = target_arrays
        _predict(
            model,
            temp_kwargs,
            batch_size=batch_size,
        )

    # Get dataset writer for the average of predictions from x, y, and z orthogonal planes
    # TODO: Skip loading raw data
    dataset_writer = CellMapDatasetWriter(**dataset_writer_kwargs)

    # Load the images for the individual predictions
    single_axis_images = {
        array_name: {
            label: [
                CellMapImage(
                    os.path.join(tmp_dir.name, "output.zarr", str(axis), label),
                    target_class=label,
                    target_scale=array_info["scale"],
                    target_voxel_shape=array_info["shape"],
                    pad=True,
                    pad_value=0,
                )
                for axis in range(3)
            ]
            for label in dataset_writer_kwargs["classes"]
        }
        for array_name, array_info in dataset_writer_kwargs["target_arrays"].items()
    }

    # Combine the predictions from the x, y, and z orthogonal planes
    print("Combining predictions.")
    for batch in tqdm(dataset_writer.loader(batch_size=batch_size), dynamic_ncols=True):
        # For each class, get the predictions from the x, y, and z orthogonal planes
        outputs = {}
        for array_name, images in single_axis_images.items():
            outputs[array_name] = {}
            for label in dataset_writer_kwargs["classes"]:
                outputs[array_name][label] = []
                for idx in batch["idx"]:
                    average_prediction = []
                    for image in images[label]:
                        average_prediction.append(image[dataset_writer.get_center(idx)])
                    average_prediction = torch.stack(average_prediction).mean(dim=0)
                    outputs[array_name][label].append(average_prediction)
                outputs[array_name][label] = torch.stack(outputs[array_name][label])

        # Save the outputs
        dataset_writer[batch["idx"]] = outputs

    tmp_dir.cleanup()


def _predict(
    model: torch.nn.Module, dataset_writer_kwargs: dict[str, Any], batch_size: int
):
    """
    Predicts the output of a model on a large dataset by splitting it into blocks and predicting each block separately.

    Parameters
    ----------
    model : torch.nn.Module
        The model to use for prediction.
    dataset_writer_kwargs : dict[str, Any]
        A dictionary containing the arguments for the dataset writer.
    batch_size : int
        The batch size to use for prediction
    """

    model.eval()
    device = dataset_writer_kwargs["device"]
    input_keys = list(dataset_writer_kwargs["input_arrays"].keys())

    if "classes" not in dataset_writer_kwargs or not dataset_writer_kwargs["classes"]:
        raise ValueError("No classes specified in dataset_writer_kwargs")
    # Get the classes to use for model output (all classes the model was trained on)
    # vs the classes to actually save (filtered by test_crop_manifest)
    model_classes = dataset_writer_kwargs.get(
        "model_classes", dataset_writer_kwargs["classes"]
    )
    # Restrict classes_to_save to only those the model knows about
    classes_to_save = [
        c for c in dataset_writer_kwargs["classes"] if c in model_classes
    ]
    dataset_writer_kwargs["classes"] = classes_to_save

    # Validate that classes_to_save is not empty
    if not classes_to_save:
        print("classes_to_save is empty. Nothing to predict. Skipping.")
        return

    # Create a mapping from class names to indices for efficient lookup during filtering
    model_class_to_index = (
        {c: i for i, c in enumerate(model_classes)}
        if model_classes != classes_to_save
        else None
    )

    # Test a single batch to get number of output channels
    test_batch = {
        k: torch.rand((1, *info["shape"])).unsqueeze(0).to(device)
        for k, info in dataset_writer_kwargs["input_arrays"].items()
    }
    test_inputs = get_data_from_batch(test_batch, input_keys, device)
    # Apply the same singleton-dimension squeezing as in the main prediction loop
    singleton_dim = get_singleton_dim(
        list(dataset_writer_kwargs["input_arrays"].values())[0]["shape"]
    )
    if singleton_dim is not None:
        test_inputs = squeeze_singleton_dim(test_inputs, singleton_dim + 2)
    with torch.no_grad():
        test_outputs = model(test_inputs)
    model_returns_class_dict = False
    num_channels_per_class = None
    if isinstance(test_outputs, dict):
        if set(test_outputs.keys()) == set(model_classes):
            # Keys are the class names; values are already per-class tensors
            model_returns_class_dict = True
        else:
            # Dict with non-class keys (e.g., resolution levels): use the first
            # value tensor to detect the channel count
            test_outputs = next(iter(test_outputs.values()))
    if not model_returns_class_dict and test_outputs.shape[1] > len(model_classes):
        if test_outputs.shape[1] % len(model_classes) == 0:
            num_channels_per_class = test_outputs.shape[1] // len(model_classes)
            # To avoid mutating the input dictionary (which may be shared across multiple
            # prediction calls), create a deep copy of target_arrays and update the shape
            # to include the channel dimension.
            target_arrays_copy = copy.deepcopy(dataset_writer_kwargs["target_arrays"])
            for key in target_arrays_copy.keys():
                current_shape = target_arrays_copy[key]["shape"]
                # Use the first input array's shape to determine expected spatial rank
                # (all input arrays should have the same spatial dimensions)
                first_input_key = next(iter(dataset_writer_kwargs["input_arrays"]))
                expected_spatial_rank = len(
                    dataset_writer_kwargs["input_arrays"][first_input_key]["shape"]
                )
                # Only prepend the channel dimension if the shape doesn't already include it
                # We check if the current rank matches the expected spatial rank (no channel dim yet)
                if len(current_shape) == expected_spatial_rank:
                    target_arrays_copy[key]["shape"] = (
                        num_channels_per_class,
                        *current_shape,
                    )
            # Replace target_arrays in the kwargs with the modified copy
            dataset_writer_kwargs = {
                **dataset_writer_kwargs,
                "target_arrays": target_arrays_copy,
            }
        else:
            raise ValueError(
                f"Number of output channels ({test_outputs.shape[1]}) does not match number of "
                f"classes ({len(model_classes)}). Should be a multiple of the "
                "number of classes."
            )
    del test_batch, test_inputs, test_outputs

    if "raw_value_transforms" not in dataset_writer_kwargs:
        dataset_writer_kwargs["raw_value_transforms"] = T.Compose(
            [
                T.ToDtype(torch.float, scale=True),
                NaNtoNum({"nan": 0, "posinf": None, "neginf": None}),
            ],
        )

    dataset_writer_kwargs = {
        k: v for k, v in dataset_writer_kwargs.items() if k != "model_classes"
    }
    dataset_writer = CellMapDatasetWriter(**dataset_writer_kwargs)
    owned_tiles = _get_owned_prediction_tiles(dataset_writer)
    for class_writers in dataset_writer.target_array_writers.values():
        for writer in class_writers.values():
            writer._zarr_array[:] = 0

    # Find singleton dimension if there is one
    # Only the first singleton dimension will be used for squeezing/unsqueezing.
    # If there are multiple singleton dimensions, only the first is handled.
    with torch.no_grad():
        tile_batches = [
            owned_tiles[start : start + batch_size]
            for start in range(0, len(owned_tiles), batch_size)
        ]
        for tiles in tqdm(tile_batches, dynamic_ncols=True):
            batch = _load_prediction_tile_batch(dataset_writer, tiles)
            # Get the inputs, handling dict vs. tensor data
            inputs = get_data_from_batch(batch, input_keys, device)
            if singleton_dim is not None:
                inputs = squeeze_singleton_dim(inputs, singleton_dim + 2)
            outputs = model(inputs)
            if singleton_dim is not None:
                outputs = unsqueeze_singleton_dim(outputs, singleton_dim + 2)

            outputs = structure_model_output(
                outputs,
                model_classes,
                num_channels_per_class,
            )

            # Filter outputs to only include the classes that should be saved
            if model_class_to_index is not None:
                filtered_outputs = {}
                for array_name, class_outputs in outputs.items():
                    if isinstance(class_outputs, dict):
                        # Filter to only include classes_to_save
                        filtered_outputs[array_name] = {
                            class_name: class_tensor
                            for class_name, class_tensor in class_outputs.items()
                            if class_name in classes_to_save
                        }
                    else:
                        # If it's not a dict (just a tensor), we need to index the tensor
                        # This assumes the tensor has shape (B, C, ...) where C corresponds to model_classes
                        # We need to select only the channels for classes_to_save
                        # classes_to_save should be a subset of model_classes by design
                        # Use pre-computed mapping for O(1) lookup instead of O(n) index()
                        class_indices = [
                            model_class_to_index[c] for c in classes_to_save
                        ]
                        filtered_outputs[array_name] = class_outputs[
                            :, class_indices, ...
                        ]
                outputs = filtered_outputs

            # Each patch writes only its disjoint responsibility region. The
            # remaining patch margin is context and is discarded.
            for batch_index, tile in enumerate(tiles):
                item_outputs = _select_prediction_batch_item(outputs, batch_index)
                _write_owned_prediction_tile(
                    dataset_writer,
                    tile,
                    item_outputs,
                )


def _estimate_output_bytes(
    target_bounds: dict[str, dict[str, list]],
    target_arrays: dict[str, dict],
    num_classes: int,
    bytes_per_voxel: int = 4,
) -> int:
    """Estimate total output size in bytes from target bounds, array scales, and class count."""
    total = 0
    axis_to_index = {"z": 0, "y": 1, "x": 2}
    for array_name, bounds in target_bounds.items():
        scale = target_arrays.get(array_name, {}).get("scale", (1, 1, 1))
        num_voxels = 1
        for axis, (lo, hi) in bounds.items():
            extent_world = hi - lo
            if axis not in axis_to_index:
                raise ValueError(
                    f"Unexpected axis {axis!r} in target bounds for array "
                    f"{array_name!r}. Expected only spatial axes 'z', 'y', or 'x'."
                )
            axis_idx = axis_to_index[axis]
            voxel_size = scale[axis_idx] if axis_idx < len(scale) else 1
            if voxel_size <= 0:
                raise ValueError(
                    f"Non-positive voxel size {voxel_size!r} for axis {axis!r} "
                    f"in target array {array_name!r}."
                )
            num_voxels *= max(1, int(extent_world / voxel_size))
        total += num_voxels * num_classes * bytes_per_voxel
    return total


def _warn_output_size(dataset_writer_kwargs: dict[str, Any]) -> None:
    """Print class list and warn if estimated output size exceeds 10 GB."""
    crop_classes = dataset_writer_kwargs.get("classes", [])
    target_path = dataset_writer_kwargs.get("target_path", "unknown")
    tqdm.write(
        f"  Crop {target_path}: saving {len(crop_classes)} classes: {crop_classes}"
    )

    target_bounds = dataset_writer_kwargs.get("target_bounds", {})
    target_arrays = dataset_writer_kwargs.get("target_arrays", {})
    if target_bounds and target_arrays:
        est_bytes = _estimate_output_bytes(
            target_bounds, target_arrays, len(crop_classes)
        )
        est_gb = est_bytes / (1024**3)
        if est_gb > 10:
            tqdm.write(
                f"  Warning: {target_path} will save {len(crop_classes)} classes and "
                f"may require approximately {est_gb:.1f} GB on disk. "
                f'Consider using crops="test" or specifying a smaller set of classes.'
            )


def predict(
    config_path: str,
    crops: str = "test",
    output_path: str = PREDICTIONS_PATH,
    do_orthoplanes: bool = True,
    overwrite: bool = False,
    search_path: str = SEARCH_PATH,
    raw_name: str = RAW_NAME,
    crop_name: str = CROP_NAME,
    filter_classes: bool = True,
):
    """
    Given a model configuration file and list of crop numbers, predicts the output of a model on a large dataset by splitting it into blocks and predicting each block separately.

    Parameters
    ----------
    config_path : str
        The path to the model configuration file. This can be the same as the config file used for training.
    crops: str, optional
        A comma-separated list of crop numbers to predict on, or "test" to predict on the entire test set. Default is "test".
        When crops="test", only the labels specified in the test_crop_manifest for each crop will be saved.
        If a crop's test_crop_manifest specifies labels that the model wasn't trained on, those labels will be
        automatically filtered out (i.e., only the intersection of model classes and crop labels will be saved).
    output_path: str, optional
        The path to save the output predictions to, formatted as a string with a placeholders for the dataset, crop number, and label. Default is PREDICTIONS_PATH set in `cellmap-segmentation/config.py`.
    do_orthoplanes: bool, optional
        Whether to compute the average of predictions from x, y, and z orthogonal planes for the full 3D volume. This is sometimes called 2.5D predictions. It expects a model that yields 2D outputs. Similarly, it expects the input shape to the model to be 2D. Default is True for 2D models.
    overwrite: bool, optional
        Whether to overwrite the output dataset if it already exists. Default is False.
    search_path: str, optional
        The path to search for the raw dataset, with placeholders for dataset and name. Default is SEARCH_PATH set in `cellmap-segmentation/config.py`.
    raw_name: str, optional
        The name of the raw dataset. Default is RAW_NAME set in `cellmap-segmentation/config.py`.
    crop_name: str, optional
        The name of the crop dataset with placeholders for crop and label. Default is CROP_NAME set in `cellmap-segmentation/config.py`.
    filter_classes: bool, optional
        When True and crops are specified by numeric ID, filter the saved classes to only those
        listed in the test_crop_manifest for each crop (intersected with the model's classes).
        When False, all model classes are saved for numeric crops. Default is True.

    Notes
    -----
    When crops="test", the function will only save predictions for labels that are specified
    in the test_crop_manifest for each specific crop AND that the model was trained on (the
    intersection of both sets). This ensures that only the labels that will be scored are saved,
    reducing storage requirements and processing time.
    """
    config = load_safe_config(config_path)
    classes = config.classes
    batch_size = getattr(config, "batch_size", 8)
    input_array_info = getattr(
        config, "input_array_info", {"shape": (1, 128, 128), "scale": (8, 8, 8)}
    )
    target_array_info = getattr(config, "target_array_info", input_array_info)
    value_transforms = getattr(
        config,
        "value_transforms",
        T.Compose(
            [
                T.ToDtype(torch.float, scale=True),
                NaNtoNum({"nan": 0, "posinf": None, "neginf": None}),
            ],
        ),
    )
    model = config.model

    # %% Check that the GPU is available
    if getattr(config, "device", None) is not None:
        device = config.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Prediction device: {device}")

    # %% Move model to device
    model = model.to(device)

    # Optionally, load a pre-trained model
    checkpoint_epoch = get_model(config)
    if checkpoint_epoch is not None:
        print(f"Loaded model checkpoint from epoch: {checkpoint_epoch}")

    if do_orthoplanes and (
        array_has_singleton_dim(input_array_info)
        or is_array_2D(input_array_info, summary=any)
    ):
        # If the model is a 2D model, compute the average of predictions from x, y, and z orthogonal planes
        predict_func = predict_orthoplanes
    elif is_array_2D(input_array_info, summary=any) or is_array_2D(
        target_array_info, summary=any
    ):
        if is_array_2D(input_array_info, summary=any):
            permute_singleton_dimension(input_array_info, axis=0)
        if is_array_2D(target_array_info, summary=any):
            permute_singleton_dimension(target_array_info, axis=0)
        print(
            "Warning: Model appears to be 2D, but do_orthoplanes is set to False. Predictions will be made only on z slices."
        )
        predict_func = _predict
    else:
        predict_func = _predict

    assert (
        input_array_info is not None and target_array_info is not None
    ), "No array info provided"
    input_arrays = {"input": input_array_info}
    target_arrays = {"output": target_array_info}

    # Get the crops to predict on
    if crops == "test":
        test_crops = get_test_crops()
        dataset_writers = []
        for crop in test_crops:
            # Get path to raw dataset
            raw_path = search_path.format(dataset=crop.dataset, name=raw_name)

            # Get the boundaries of the crop
            target_bounds = {
                "output": {
                    axis: [
                        crop.gt_source.translation[i],
                        crop.gt_source.translation[i]
                        + crop.gt_source.voxel_size[i] * crop.gt_source.shape[i],
                    ]
                    for i, axis in enumerate("zyx")
                },
            }

            # Get the labels that should be scored for this specific crop from the test_crop_manifest
            crop_labels = get_test_crop_labels(crop.id)
            # Filter to only include labels that are in the model's classes
            filtered_classes = [c for c in classes if c in crop_labels]

            # If there are no matching labels between the model and this crop, skip it
            if not filtered_classes:
                tqdm.write(
                    f"Skipping crop {crop.id} (dataset={crop.dataset}) because there are "
                    f"no labels in common between model classes {classes} and crop labels {crop_labels}."
                )
                continue

            # Create the writer
            # Note: We pass all classes to the model for prediction, but only the filtered
            # classes will be saved by the CellMapDatasetWriter
            dataset_writers.append(
                {
                    "raw_path": raw_path,
                    "target_path": output_path.format(
                        crop=f"crop{crop.id}",
                        dataset=crop.dataset,
                    ),
                    "classes": filtered_classes,
                    "model_classes": classes,  # All classes the model was trained on
                    "input_arrays": input_arrays,
                    "target_arrays": target_arrays,
                    "target_bounds": target_bounds,
                    "overwrite": overwrite,
                    "device": device,
                    "raw_value_transforms": value_transforms,
                }
            )
    else:
        crop_list = crops.split(",")
        crop_path_matches: dict[str, list[str]] = {}
        for i, crop in enumerate(crop_list):
            if (isinstance(crop, str) and crop.isnumeric()) or isinstance(crop, int):
                crop = f"crop{crop}"
                crop_list[i] = crop  # type: ignore

            matches = glob(
                search_path.format(
                    dataset="*", name=crop_name.format(crop=crop, label="")
                ).rstrip(os.path.sep)
            )
            if not matches:
                tqdm.write(
                    f"Warning: no input paths found for crop '{crop}', skipping."
                )
            crop_path_matches[crop] = matches

        dataset_writers = []
        for crop, crop_paths_for_crop in crop_path_matches.items():  # type: ignore
            # Optionally filter classes using the test_crop_manifest.
            # Only applies when the crop ID is numeric AND appears in the
            # test manifest; non-test crops always save all model classes.
            filtered_classes = classes
            if filter_classes:
                crop_id_str = crop.replace("crop", "")
                if crop_id_str.isnumeric():
                    crop_labels = get_test_crop_labels(int(crop_id_str))
                    if crop_labels:
                        filtered_classes = [c for c in classes if c in crop_labels]
                        if not filtered_classes:
                            tqdm.write(
                                f"Skipping {crop} because there are no labels in common "
                                f"between model classes {classes} and crop labels {crop_labels}."
                            )
                            continue

            for crop_path in crop_paths_for_crop:
                # Get path to raw dataset
                raw_path = get_raw_path(crop_path, label="")

                # Get the boundaries of the crop from every readable label.
                # Using only classes[0] is fragile when the first class is rare
                # or missing for a crop.
                target_bounds = _get_crop_target_bounds(
                    crop_path,
                    classes,
                    target_arrays,
                )

                dataset = get_formatted_fields(raw_path, search_path, ["{dataset}"])[
                    "dataset"
                ]

                # Create the writer
                writer_kwargs = {
                    "raw_path": raw_path,
                    "target_path": output_path.format(crop=crop, dataset=dataset),
                    "classes": filtered_classes,
                    "input_arrays": input_arrays,
                    "target_arrays": target_arrays,
                    "target_bounds": target_bounds,
                    "overwrite": overwrite,
                    "device": device,
                    "raw_value_transforms": value_transforms,
                }
                if filtered_classes != classes:
                    writer_kwargs["model_classes"] = classes
                dataset_writers.append(writer_kwargs)

    for dataset_writer_kwargs in dataset_writers:
        _warn_output_size(dataset_writer_kwargs)
        predict_func(model, dataset_writer_kwargs, batch_size)
