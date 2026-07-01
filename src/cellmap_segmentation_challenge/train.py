import gc
import io
import os
import random
import time

import numpy as np
import torch
import torchvision.transforms.v2 as T
from cellmap_data.utils import get_fig_dict, longest_common_substring
from cellmap_data.transforms.augment import NaNtoNum, Binarize
from tensorboardX import SummaryWriter
from tqdm import tqdm
from upath import UPath
import matplotlib.pyplot as plt

from .models import ResNet, UNet_2D, UNet_3D, ViTVNet, get_model
from .utils import (
    CellMapLossWrapper,
    get_dataloader,
    load_safe_config,
    make_datasplit_csv,
    make_s3_datasplit_csv,
    format_string,
    get_data_from_batch,
    get_singleton_dim,
    squeeze_singleton_dim,
    unsqueeze_singleton_dim,
)


def train(config_path: str):
    """
    Train a model using the configuration file at the specified path. The model checkpoints and training logs, as well as the datasets used for training, will be saved to the paths specified in the configuration file.

    Parameters
    ----------
    config_path : str
        Path to the configuration file to use for training the model. This file should be a Python file that defines the hyperparameters and other configurations for training the model. This may include:
        - base_experiment_path: Path to the base experiment directory. Default is the parent directory of the config file.
        - model_save_path: Path to save the model checkpoints. Default is 'checkpoints/{model_name}_{epoch}.pth' within the base_experiment_path.
        - logs_save_path: Path to save the logs for tensorboard. Default is 'tensorboard/{model_name}' within the base_experiment_path. Training progress may be monitored by running `tensorboard --logdir <logs_save_path>` in the terminal.
        - datasplit_path: Path to the datasplit file that defines the train/val split the dataloader should use. Default is 'datasplit.csv' within the base_experiment_path.
        - validation_prob: Proportion of the datasets to use for validation. This is used if the datasplit CSV specified by `datasplit_path` does not already exist. Default is 0.15.
        - learning_rate: Learning rate for the optimizer. Default is 0.0001.
        - batch_size: Batch size for the dataloader. Default is 8.
        - input_array_info: Dictionary containing the shape and scale of the input data. Default is {'shape': (1, 128, 128), 'scale': (8, 8, 8)}.
        - target_array_info: Dictionary containing the shape and scale of the target data. Default is to use `input_array_info`.
        - epochs: Number of epochs to train the model for. Default is 1000.
        - iterations_per_epoch: Number of iterations per epoch. Each iteration includes an independently generated random batch from the training set. Default is 1000.
        - random_seed: Random seed for reproducibility. Default is 42.
        - classes: List of classes to train the model to predict. This will be reflected in the data included in the datasplit, if generated de novo after calling this script. Default is ['nuc', 'er'].
        - model_name: Name of the model to use. If the config file constructs the PyTorch model, this name can be anything. If the config file does not construct the PyTorch model, the model_name will need to specify which included architecture to use. This includes '2d_unet', '2d_resnet', '3d_unet', '3d_resnet', and 'vitnet'. Default is '2d_unet'. See the `models` module `README.md` for more information.
        - model_to_load: Name of the pre-trained model to load. Default is the same as `model_name`.
        - model_kwargs: Dictionary of keyword arguments to pass to the model constructor. Default is {}. If the PyTorch `model` is passed, this will be ignored. See the `models` module `README.md` for more information.
        - model: PyTorch model to use for training. If this is provided, the `model_name` and `model_to_load` can be any string. Default is None.
        - load_model: Which model checkpoint to load if it exists. Options are 'latest' or 'best'. If no checkpoints exist, will silently use the already initialized model. Default is 'latest'.
        - spatial_transforms: Dictionary of spatial transformations to apply to the training data. Default is {'mirror': {'axes': {'x': 0.5, 'y': 0.5}}, 'transpose': {'axes': ['x', 'y']}, 'rotate': {'axes': {'x': [-180, 180], 'y': [-180, 180]}}}. See the `dataloader` module documentation for more information.
        - validation_time_limit: Maximum time to spend on validation in seconds. If None, there is no time limit. Default is None.
        - validation_batch_limit: Maximum number of validation batches to process. If None, there is no limit. Default is None.
        - device: Device to use for training. If None, will use 'cuda' if available, 'mps' if available, or 'cpu' otherwise. Default is None.
        - use_s3: Whether to use the S3 bucket for the datasplit. Default is False.
        - optimizer: PyTorch optimizer to use for training. Default is `torch.optim.RAdam(model.parameters(), lr=learning_rate, decoupled_weight_decay=True)`.
        - criterion: Uninstantiated PyTorch loss function to use for training. Default is `torch.nn.BCEWithLogitsLoss`.
        - criterion_kwargs: Dictionary of keyword arguments to pass to the training loss constructor. Default is {}.
        - validation_criterion: Optional loss function used only for validation. If omitted, validation uses the instantiated training criterion for backward compatibility.
        - validation_criterion_kwargs: Dictionary of keyword arguments passed to `validation_criterion`. Default is {}.
        - validation_wrap_loss: Whether to wrap `validation_criterion` with `CellMapLossWrapper`. Default is the value of `wrap_loss`.
        - weight_loss: Whether to weight the loss function by class counts found in the datasets. Default is True.
        - use_mutual_exclusion: Whether to use mutual exclusion to infer labels for unannotated pixels. Default is False.
        - weighted_sampler: Whether to use a sampler weighted by class counts for the dataloader. Default is True.
        - train_raw_value_transforms: Transform to apply to the raw values for training. Defaults to T.Compose([T.ToDtype(torch.float, scale=True), NaNtoNum({"nan": 0, "posinf": None, "neginf": None})]) which normalizes the input data, converts it to float32, and replaces NaNs with 0. This can be used to add augmentations such as random erasing, blur, noise, etc.
        - val_raw_value_transforms: Transform to apply to the raw values for validation, similar to `train_raw_value_transforms`. Default is the same as `train_raw_value_transforms`.
        - target_value_transforms: Transform to apply to the target values. Default is T.Compose([T.ToDtype(torch.float), Binarize()]) which converts the input masks to float32 and threshold at 0 (turning object ID's into binary masks for use with binary cross entropy loss). This can be used to specify other targets, such as distance transforms.
        - max_grad_norm: Maximum gradient norm for clipping. If None, no clipping is performed. Default is None. This can be useful to prevent exploding gradients which would lead to NaNs in the weights.
        - force_all_classes: Whether to force all classes to be present in each batch provided by dataloaders. Can either be `True` to force this for both validation and training dataloader, `False` to force for neither, or `train` / `validate` to restrict it to training or validation, respectively. Default is 'validate'.
        - scheduler: PyTorch learning rate scheduler (or uninstantiated class) to use for training. Default is None. If provided, the scheduler will be called at the end of each epoch.
        - scheduler_kwargs: Dictionary of keyword arguments to pass to the scheduler constructor. Default is {}. If `scheduler` instantiation is provided, this will be ignored.
        - filter_by_scale: Whether to filter the data by scale. If True, only data with a scale less than or equal to the `input_array_info` highest resolution will be included in the datasplit. If set to a scalar value, data will be filtered for that isotropic resolution - anisotropic can be specified with a sequence of scalars. Default is False (no filtering).
        - gradient_accumulation_steps: Number of gradient accumulation steps to use. Default is 1. This can be used to simulate larger batch sizes without increasing memory usage.
        - dataloader_kwargs: Additional keyword arguments to pass to the CellMapDataLoader. Default is {}.
        - debug_memory: Whether to enable memory debugging using `objgraph`. When True, object-growth statistics are printed before the training loop (as a baseline) and then periodically during training. Requires `pip install objgraph`; if the package is not found, a warning is printed and the flag is silently ignored. Default is False. The logging interval is controlled by the ``MEMORY_LOG_STEPS`` environment variable (default 100 iterations).

    Returns
    -------
    None

    """
    # Pick the fastest deterministic algorithm for the hardware
    torch.backends.cudnn.deterministic = True

    # Helper functions for memory management
    def _clone_tensors(obj):
        """Recursively clone tensors in nested structures (dicts, lists, tuples)."""
        if isinstance(obj, dict):
            return {k: _clone_tensors(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            cloned_list = [_clone_tensors(v) for v in obj]
            return type(obj)(cloned_list)
        return obj.detach().clone() if torch.is_tensor(obj) else obj

    def _clear_memory(force_gc=False):
        """Clear GPU cache and optionally trigger garbage collection."""
        if force_gc:
            gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _save_training_batch_for_viz(batch, inputs, outputs, targets):
        """Save a clone of the last training batch for visualization fallback."""
        return (
            _clone_tensors(batch),
            _clone_tensors(inputs),
            _clone_tensors(outputs),
            _clone_tensors(targets),
        )

    def _take_batch_indices(data, indices):
        """Take selected items along the batch dimension from nested tensors."""
        if torch.is_tensor(data):
            return data.index_select(0, indices)
        if isinstance(data, dict):
            return {
                key: _take_batch_indices(value, indices)
                for key, value in data.items()
            }
        if isinstance(data, (list, tuple)):
            return type(data)(_take_batch_indices(value, indices) for value in data)
        return data

    def _concat_batch_parts(parts):
        """Concatenate nested tensors along the batch dimension."""
        if len(parts) == 1:
            return parts[0]
        first = parts[0]
        if torch.is_tensor(first):
            return torch.cat(parts, dim=0)
        if isinstance(first, dict):
            return {
                key: _concat_batch_parts([part[key] for part in parts])
                for key in first
            }
        if isinstance(first, (list, tuple)):
            return type(first)(
                _concat_batch_parts([part[index] for part in parts])
                for index in range(len(first))
            )
        return first

    def _batch_length(data):
        if torch.is_tensor(data):
            return data.shape[0]
        if isinstance(data, dict):
            return _batch_length(next(iter(data.values())))
        if isinstance(data, (list, tuple)):
            return _batch_length(data[0])
        raise ValueError("Cannot infer batch length for patch filtering.")

    def _target_ratio_keep_indices(targets, class_indices, threshold):
        """Return batch indices where selected class channels stay under threshold."""
        if isinstance(targets, dict):
            if len(targets) != 1:
                raise ValueError(
                    "patch_filter currently expects a single target tensor, "
                    f"but got {len(targets)} target arrays."
                )
            targets = next(iter(targets.values()))
        if not torch.is_tensor(targets) or targets.ndim < 3:
            raise ValueError(
                "patch_filter expects targets with shape (B, C, ...)."
            )

        channels = targets.shape[1]
        bad_indices = [
            index for index in class_indices if index < 0 or index >= channels
        ]
        if bad_indices:
            raise ValueError(
                f"patch_filter_class_indices {bad_indices} are outside target "
                f"channel range 0..{channels - 1}."
            )

        finite_targets = targets.isfinite()
        valid_voxels = finite_targets.any(dim=1)
        valid_counts = valid_voxels.flatten(1).sum(dim=1)

        selected = targets[:, list(class_indices)].nan_to_num(0)
        selected_voxels = selected.sum(dim=1) > 0.5
        selected_counts = (selected_voxels & valid_voxels).flatten(1).sum(dim=1)

        ratios = torch.zeros(
            targets.shape[0],
            device=targets.device,
            dtype=torch.float32,
        )
        non_empty = valid_counts > 0
        ratios[non_empty] = (
            selected_counts[non_empty].float() / valid_counts[non_empty].float()
        )
        keep = non_empty & (ratios <= threshold)
        return torch.where(keep)[0], ratios

    def _load_batch_tensors(batch):
        inputs = get_data_from_batch(batch, input_keys, device)
        if singleton_dim is not None:
            inputs = squeeze_singleton_dim(inputs, singleton_dim + 2)
        targets = get_data_from_batch(batch, target_keys, device)
        return inputs, targets

    def _next_loader_batch(loader_iter):
        try:
            return next(loader_iter), loader_iter
        except StopIteration:
            loader_iter = iter(train_loader.loader)
            return next(loader_iter), loader_iter

    def _collect_filtered_training_batch(loader_iter):
        """Collect valid patches before model forward according to config."""
        input_parts = []
        target_parts = []
        raw_batch_for_viz = None
        valid_count = 0
        attempts = 0
        rejected_count = 0

        while valid_count < batch_size and attempts < patch_filter_max_attempts:
            raw_batch, loader_iter = _next_loader_batch(loader_iter)
            attempts += 1
            inputs, targets = _load_batch_tensors(raw_batch)
            keep_indices, _ = _target_ratio_keep_indices(
                targets,
                patch_filter_class_indices,
                patch_filter_ratio_threshold,
            )
            rejected_count += int(_batch_length(targets) - keep_indices.numel())

            if keep_indices.numel() > 0:
                need = batch_size - valid_count
                keep_indices = keep_indices[:need]
                input_parts.append(_take_batch_indices(inputs, keep_indices))
                target_parts.append(_take_batch_indices(targets, keep_indices))
                valid_count += keep_indices.numel()
                if raw_batch_for_viz is None:
                    raw_batch_for_viz = raw_batch

            if keep_indices.numel() == 0:
                del inputs, targets, raw_batch

        if valid_count < patch_filter_min_batch_size:
            return None, None, None, loader_iter, attempts, rejected_count

        inputs = _concat_batch_parts(input_parts)
        targets = _concat_batch_parts(target_parts)
        return (
            raw_batch_for_viz,
            inputs,
            targets,
            loader_iter,
            attempts,
            rejected_count,
        )

    def _foreground_targets_for_viz(target_data):
        """Drop auxiliary target channels that the model does not predict."""
        if torch.is_tensor(target_data) and target_data.ndim >= 2:
            return target_data[:, : len(classes)]
        return target_data

    # %% Load the configuration file
    config = load_safe_config(config_path)
    # %% Set hyperparameters and other configurations from the config file
    base_experiment_path = getattr(
        config, "base_experiment_path", UPath(config_path).parent
    )
    base_experiment_path = UPath(base_experiment_path)
    model_save_path = getattr(
        config,
        "model_save_path",
        (base_experiment_path / "checkpoints" / "{model_name}_{epoch}.pth").path,
    )
    logs_save_path = getattr(
        config,
        "logs_save_path",
        (base_experiment_path / "tensorboard" / "{model_name}").path,
    )
    datasplit_path = getattr(
        config, "datasplit_path", (base_experiment_path / "datasplit.csv").path
    )
    validation_prob = getattr(config, "validation_prob", 0.15)
    learning_rate = getattr(config, "learning_rate", 0.0001)
    batch_size = getattr(config, "batch_size", 8)
    filter_by_scale = getattr(config, "filter_by_scale", False)
    input_array_info = getattr(
        config, "input_array_info", {"shape": (1, 128, 128), "scale": (8, 8, 8)}
    )
    target_array_info = getattr(config, "target_array_info", input_array_info)
    epochs = getattr(config, "epochs", 1000)
    iterations_per_epoch = getattr(config, "iterations_per_epoch", 1000)
    random_seed = getattr(config, "random_seed", getattr(os.environ, "SEED", 42))
    classes = getattr(config, "classes", ["nuc", "er"])
    target_classes = getattr(config, "target_classes", classes)
    model_name = getattr(config, "model_name", "2d_unet")
    model_to_load = getattr(config, "model_to_load", model_name)
    model_kwargs = getattr(config, "model_kwargs", {})
    model = getattr(config, "model", None)
    spatial_transforms = getattr(
        config,
        "spatial_transforms",
        {
            "mirror": {"axes": {"x": 0.5, "y": 0.5}},
            "transpose": {"axes": ["x", "y"]},
            "rotate": {"axes": {"x": [-180, 180], "y": [-180, 180]}},
        },
    )
    validation_time_limit = getattr(config, "validation_time_limit", None)
    validation_batch_limit = getattr(config, "validation_batch_limit", None)
    device = getattr(config, "device", None)
    use_s3 = getattr(config, "use_s3", False)
    use_mutual_exclusion = getattr(config, "use_mutual_exclusion", False)
    weighted_sampler = getattr(config, "weighted_sampler", True)
    train_raw_value_transforms = getattr(
        config,
        "train_raw_value_transforms",
        T.Compose(
            [
                T.ToDtype(torch.float, scale=True),
                NaNtoNum({"nan": 0, "posinf": None, "neginf": None}),
            ],
        ),
    )
    val_raw_value_transforms = getattr(
        config,
        "val_raw_value_transforms",
        T.Compose(
            [
                T.ToDtype(torch.float, scale=True),
                NaNtoNum({"nan": 0, "posinf": None, "neginf": None}),
            ],
        ),
    )
    target_value_transforms = getattr(
        config,
        "target_value_transforms",
        T.Compose([T.ToDtype(torch.float, scale=True), Binarize(0.5)]),
    )
    max_grad_norm = getattr(config, "max_grad_norm", None)
    force_all_classes = getattr(config, "force_all_classes", "validate")
    print("force_all_classes:", force_all_classes)
    patch_filter_class_indices = getattr(config, "patch_filter_class_indices", None)
    patch_filter_ratio_threshold = getattr(
        config,
        "patch_filter_ratio_threshold",
        None,
    )
    patch_filter_max_attempts = getattr(config, "patch_filter_max_attempts", 50)
    patch_filter_min_batch_size = getattr(config, "patch_filter_min_batch_size", 1)
    use_patch_filter = (
        patch_filter_class_indices is not None
        and patch_filter_ratio_threshold is not None
    )
    if use_patch_filter:
        patch_filter_class_indices = tuple(patch_filter_class_indices)
        if not 0.0 <= patch_filter_ratio_threshold <= 1.0:
            raise ValueError("patch_filter_ratio_threshold must be between 0 and 1")
        if patch_filter_max_attempts < 1:
            raise ValueError("patch_filter_max_attempts must be >= 1")
        if patch_filter_min_batch_size < 1:
            raise ValueError("patch_filter_min_batch_size must be >= 1")
        print(
            "Patch filter enabled: keeping patches with selected-class ratio "
            f"<= {patch_filter_ratio_threshold}; class indices="
            f"{patch_filter_class_indices}; target batch_size={batch_size}."
        )

    # %% Define the optimizer, from the config file or default to RAdam
    optimizer = getattr(
        config,
        "optimizer",
        torch.optim.RAdam(
            model.parameters(), lr=learning_rate, decoupled_weight_decay=True
        ),
    )

    # %% Define the scheduler, from the config file or default to None
    scheduler = getattr(config, "scheduler", None)
    if isinstance(scheduler, type):
        scheduler_kwargs = getattr(config, "scheduler_kwargs", {})
        scheduler = scheduler(optimizer, **scheduler_kwargs)

    # %% Define the loss function, from the config file or default to BCEWithLogitsLoss
    criterion = getattr(config, "criterion", torch.nn.BCEWithLogitsLoss)
    criterion_kwargs = dict(getattr(config, "criterion_kwargs", {}))
    weight_loss = getattr(config, "weight_loss", True)
    wrap_loss = getattr(config, "wrap_loss", True)
    configured_validation_criterion = getattr(config, "validation_criterion", None)
    validation_criterion_kwargs = dict(
        getattr(config, "validation_criterion_kwargs", {})
    )
    validation_wrap_loss = getattr(config, "validation_wrap_loss", wrap_loss)

    gradient_accumulation_steps = getattr(
        config, "gradient_accumulation_steps", 1
    )  # default to 1 for no accumulation
    if gradient_accumulation_steps < 1:
        raise ValueError(
            f"gradient_accumulation_steps must be >= 1, but got {gradient_accumulation_steps}"
        )

    # %% Make sure the save path exists
    for path in [model_save_path, logs_save_path, datasplit_path]:
        dirpath = os.path.dirname(path)
        if len(dirpath) > 0:
            os.makedirs(dirpath, exist_ok=True)

    # %% Set the random seed
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)

    # %% Check that the GPU is available
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    print(f"Training device: {device}")

    # %% Make the datasplit file if it doesn't exist
    if not os.path.exists(datasplit_path):
        if filter_by_scale is not False:
            # Find highest resolution scale
            if filter_by_scale is not True:
                scale = filter_by_scale
                if isinstance(scale, (int, float)):
                    scale = (scale, scale, scale)
            elif "scale" in input_array_info:
                scale = input_array_info["scale"]
            else:
                highest_res = [np.inf, np.inf, np.inf]
                for key, info in input_array_info.items():
                    if "scale" in info:
                        res = np.prod(info["scale"])
                        if res < np.prod(highest_res):
                            highest_res = info["scale"]
                scale = highest_res
        else:
            scale = None
        # Make the datasplit CSV
        if use_s3:
            make_s3_datasplit_csv(
                classes=target_classes,
                scale=scale,
                csv_path=datasplit_path,
                validation_prob=validation_prob,
                force_all_classes=force_all_classes,
            )
        else:
            make_datasplit_csv(
                classes=target_classes,
                scale=scale,
                csv_path=datasplit_path,
                validation_prob=validation_prob,
                force_all_classes=force_all_classes,
            )

    # %% Download the data and make the dataloader
    train_loader, val_loader = get_dataloader(
        datasplit_path=datasplit_path,
        classes=target_classes,
        batch_size=batch_size,
        input_array_info=input_array_info,
        target_array_info=target_array_info,
        spatial_transforms=spatial_transforms,
        iterations_per_epoch=iterations_per_epoch,
        random_validation=validation_time_limit or validation_batch_limit,
        device=device,
        weighted_sampler=weighted_sampler,
        use_mutual_exclusion=use_mutual_exclusion,
        train_raw_value_transforms=train_raw_value_transforms,
        val_raw_value_transforms=val_raw_value_transforms,
        target_value_transforms=target_value_transforms,
        **config.get("dataloader_kwargs", {}),
    )

    # %% If no model is provided, create a new model
    if model is None:
        if "2d" in model_name.lower():
            if "unet" in model_name.lower():
                model = UNet_2D(1, len(classes), **model_kwargs)
            elif "resnet" in model_name.lower():
                model = ResNet(ndims=2, output_nc=len(classes), **model_kwargs)
            else:
                raise ValueError(
                    f"Unknown model name: {model_name}. Preconfigured 2D models are '2d_unet' and '2d_resnet'."
                )
        elif "3d" in model_name.lower():
            if "unet" in model_name.lower():
                model = UNet_3D(1, len(classes), **model_kwargs)
            elif "resnet" in model_name.lower():
                model = ResNet(ndims=3, output_nc=len(classes), **model_kwargs)
            else:
                raise ValueError(
                    f"Unknown model name: {model_name}. Preconfigured 3D models are '3d_unet' and '3d_resnet', or 'vitnet'."
                )
        elif "vitnet" in model_name.lower():
            model = ViTVNet(len(classes), **model_kwargs)
        else:
            raise ValueError(
                f"Unknown model name: {model_name}. Preconfigured models are '2d_unet', '2d_resnet', '3d_unet', '3d_resnet', and 'vitnet'. Otherwise provide a custom model as a torch.nn.Module."
            )

    # Optionally, load a pre-trained model
    checkpoint_epoch = get_model(config)

    # Create a variables for tracking iterations and offseting already completed epochs, if applicable
    epochs = np.arange(epochs) + 1
    if model_to_load == model_name:
        # Calculate the number of iterations to skip
        # NOTE: This assumes that the number of iterations per epoch is consistent
        n_iter = checkpoint_epoch * iterations_per_epoch
        epochs += checkpoint_epoch
    else:
        n_iter = 0

    # %% Move model to device
    model = model.to(device)

    # Deduce number of spatial dimensions
    if "shape" in target_array_info:
        spatial_dims = sum([s > 1 for s in target_array_info["shape"]])
    else:
        spatial_dims = sum(
            [s > 1 for s in list(target_array_info.values())[0]["shape"]]
        )

    # Use custom loss function wrapper that handles NaN values in the target.
    # Multi-class losses can opt out because their target shape differs from BCE.
    if weight_loss and wrap_loss:
        pos_weight = list(train_loader.dataset.class_weights.values())
        pos_weight = torch.tensor(pos_weight, dtype=torch.float32).to(device).flatten()

        # Adjust weight for data dimensions
        pos_weight = pos_weight[:, None, None]
        if spatial_dims == 3:
            pos_weight = pos_weight[..., None]
        criterion_kwargs["pos_weight"] = pos_weight
    if wrap_loss:
        train_criterion = CellMapLossWrapper(criterion, **criterion_kwargs)
    elif isinstance(criterion, type):
        train_criterion = criterion(**criterion_kwargs)
    else:
        train_criterion = criterion

    if configured_validation_criterion is None:
        validation_criterion = train_criterion
    elif validation_wrap_loss:
        validation_criterion = CellMapLossWrapper(
            configured_validation_criterion,
            **validation_criterion_kwargs,
        )
    elif isinstance(configured_validation_criterion, type):
        validation_criterion = configured_validation_criterion(
            **validation_criterion_kwargs
        )
    else:
        validation_criterion = configured_validation_criterion

    input_keys = list(train_loader.dataset.input_arrays.keys())
    target_keys = list(train_loader.dataset.target_arrays.keys())

    # Find singleton dimension for 2D-model squeeze/unsqueeze
    if "shape" in input_array_info:
        first_input_shape = input_array_info["shape"]
    else:
        first_input_shape = list(input_array_info.values())[0]["shape"]
    singleton_dim = get_singleton_dim(first_input_shape)

    # %% Train the model
    post_fix_dict = {}

    # Define a summarywriter
    writer = SummaryWriter(format_string(logs_save_path, {"model_name": model_name}))

    # Training outer loop, across epochs
    print(
        f"Training {model_name} for {len(epochs)} epochs, starting at epoch {epochs[0]}, iteration {n_iter}..."
    )

    debug_memory = getattr(config, "debug_memory", False)
    memory_log_steps = int(os.environ.get("MEMORY_LOG_STEPS", "100"))
    if debug_memory:
        try:
            import objgraph
        except ImportError:
            print(
                "objgraph is not installed. Install it with `pip install objgraph` to enable memory debugging features."
            )
            debug_memory = False
        else:
            # Before training loop
            objgraph.show_growth(limit=5, file=io.StringIO())  # establish baseline

    for epoch in epochs:

        # Set the model to training mode to enable backpropagation
        model.train()

        # Training loop for the epoch
        post_fix_dict["Epoch"] = epoch

        train_iter = iter(train_loader.loader)
        epoch_bar = tqdm(
            range(iterations_per_epoch),
            desc="Training",
            dynamic_ncols=True,
            total=iterations_per_epoch,
        )
        optimizer.zero_grad()
        for epoch_iter in epoch_bar:
            # Increment the training iteration
            n_iter += 1

            # Collect a training batch. If configured, reject patches before
            # forward until a full valid batch is assembled.
            if use_patch_filter:
                (
                    batch,
                    inputs,
                    targets,
                    train_iter,
                    filter_attempts,
                    rejected_count,
                ) = _collect_filtered_training_batch(train_iter)
                if inputs is None:
                    post_fix_dict["Skipped"] = (
                        f"no valid patches in {filter_attempts} attempts"
                    )
                    epoch_bar.set_postfix(post_fix_dict)
                    optimizer.zero_grad()
                    n_iter -= 1
                    continue
                post_fix_dict["Filter"] = (
                    f"{_batch_length(inputs)}/{batch_size} kept, "
                    f"{rejected_count} rejected, {filter_attempts} draws"
                )
            else:
                batch, train_iter = _next_loader_batch(train_iter)
                inputs, targets = _load_batch_tensors(batch)

            # Forward pass (compute the model outputs)
            outputs = model(inputs)
            if singleton_dim is not None:
                outputs = unsqueeze_singleton_dim(outputs, singleton_dim + 2)

            # Compute the loss
            if outputs.ndim == 5 and targets.ndim == 4 and outputs.shape[2] == 1:
                outputs = outputs.squeeze(2)

            loss = train_criterion(outputs, targets) / gradient_accumulation_steps

            # Backward pass (compute the gradients)
            loss.backward()

            # Clip the gradients if necessary
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            # Update the weights
            if (
                (epoch_iter + 1) % gradient_accumulation_steps == 0
                or epoch_iter >= iterations_per_epoch - 1
            ):
                # Only update the weights every `gradient_accumulation_steps` iterations
                # This allows for larger effective batch sizes without increasing memory usage
                optimizer.step()
                optimizer.zero_grad()

            # Update the progress bar
            post_fix_dict["Loss"] = f"{loss.item()}"
            epoch_bar.set_postfix(post_fix_dict)

            # Log the loss using tensorboard
            writer.add_scalar("loss", loss.item(), n_iter)

            # Save last batch for visualization if validation won't run
            # Only save when needed to minimize memory overhead
            if epoch_iter >= iterations_per_epoch - 1 and (
                val_loader is None or len(val_loader.loader) == 0
            ):
                (
                    last_train_batch,
                    last_train_inputs,
                    last_train_outputs,
                    last_train_targets,
                ) = _save_training_batch_for_viz(batch, inputs, outputs, targets)

            # Clean up references to free memory
            del batch, inputs, targets, outputs, loss

            # Periodically clear GPU cache to prevent memory accumulation
            if epoch_iter > 0 and epoch_iter % memory_log_steps == 0:
                if debug_memory:
                    print(f"Memory usage at iteration {n_iter}:")
                    objgraph.show_growth(limit=5)
                _clear_memory()

            # Check if we've reached the specified number of iterations for this epoch
            if epoch_iter >= iterations_per_epoch - 1:
                break

        # Trigger garbage collection to reclaim memory
        _clear_memory(force_gc=True)

        if scheduler is not None:
            # Step the scheduler at the end of each epoch
            scheduler.step()
            writer.add_scalar("lr", scheduler.get_last_lr()[0], n_iter)

        # Save the model
        torch.save(
            model.state_dict(),
            format_string(model_save_path, {"epoch": epoch, "model_name": model_name}),
        )

        # Compute the validation score by averaging the loss across the validation set
        if val_loader is not None and len(val_loader.loader) > 0:
            val_score = 0
            if validation_time_limit is not None:
                elapsed_time = 0
                last_time = time.time()
                val_bar = val_loader.loader
                pbar = tqdm(
                    total=validation_time_limit,
                    desc="Validation",
                    bar_format="{l_bar}{bar}| {remaining}s remaining",
                    dynamic_ncols=True,
                )
            else:
                val_bar = tqdm(
                    val_loader.loader,
                    desc="Validation",
                    total=validation_batch_limit or len(val_loader.loader),
                    dynamic_ncols=True,
                )

            # Free up GPU memory, disable backprop etc.
            _clear_memory()
            optimizer.zero_grad()
            model.eval()

            # Validation loop
            with torch.no_grad():
                i = 0
                for batch in val_bar:
                    inputs = get_data_from_batch(batch, input_keys, device)
                    if singleton_dim is not None:
                        inputs = squeeze_singleton_dim(inputs, singleton_dim + 2)
                    outputs = model(inputs)
                    if singleton_dim is not None:
                        outputs = unsqueeze_singleton_dim(outputs, singleton_dim + 2)

                    # Compute the loss
                    targets = get_data_from_batch(batch, target_keys, device)

                    if outputs.ndim == 5 and targets.ndim == 4 and outputs.shape[2] == 1:
                        outputs = outputs.squeeze(2)

                    val_score += validation_criterion(outputs, targets).item()
                    i += 1

                    # Check time limit
                    if validation_time_limit is not None:
                        last_elapsed_time = time.time() - last_time
                        elapsed_time += last_elapsed_time
                        if elapsed_time >= validation_time_limit:
                            break
                        pbar.update(last_elapsed_time)
                        last_time = time.time()
                    # Check batch limit
                    elif (
                        validation_batch_limit is not None
                        and i >= validation_batch_limit
                    ):
                        break
                val_score /= i

                # Log the validation using tensorboard
                writer.add_scalar("validation", val_score, n_iter)

                # Update the progress bar
                post_fix_dict["Validation"] = f"{val_score:.4f}"

            # Trigger garbage collection and clear GPU cache for memory no longer referenced
            # Note: validation batch tensors remain referenced for visualization below
            _clear_memory(force_gc=True)

        # Generate and save figures from the last batch (validation if available, otherwise training)
        # Check if validation batch variables exist; if not, try to use saved training batch
        has_batch_for_viz = False
        try:
            # Try to access validation batch variables
            _ = batch
            has_batch_for_viz = True
        except NameError:
            # Validation didn't run, try to use saved training batch if it exists
            try:
                batch = last_train_batch
                inputs = last_train_inputs
                outputs = last_train_outputs
                targets = last_train_targets
                has_batch_for_viz = True
            except NameError:
                # No batch available for visualization (training didn't reach last iteration)
                pass

        # Generate figures from the batch data if available
        if has_batch_for_viz:
            if isinstance(outputs, dict):
                outputs = list(outputs.values())
            if len(input_keys) == len(target_keys) != 1:
                # If the number of input and target keys is the same, assume they are paired
                for i, (in_key, target_key) in enumerate(zip(input_keys, target_keys)):
                    figs = get_fig_dict(
                        input_data=batch[in_key],
                        target_data=_foreground_targets_for_viz(batch[target_key]),
                        outputs=outputs[i],
                        classes=classes,
                    )
                    array_name = longest_common_substring(in_key, target_key)
                    for name, fig in figs.items():
                        writer.add_figure(f"{name}: {array_name}", fig, n_iter)
                        writer.flush()  # Ensure figures are written to disk
                        plt.close(fig)

            else:
                # If the number of input and target keys is not the same, assume that only the first input and target keys match
                if isinstance(outputs, list):
                    outputs = outputs[0]
                if isinstance(inputs, dict):
                    inputs = list(inputs.values())[0]
                elif isinstance(inputs, list):
                    inputs = inputs[0]
                if isinstance(targets, dict):
                    targets = list(targets.values())[0]
                elif isinstance(targets, list):
                    targets = targets[0]
                figs = get_fig_dict(
                    inputs,
                    _foreground_targets_for_viz(targets),
                    outputs,
                    classes,
                )
                for name, fig in figs.items():
                    writer.add_figure(name, fig, n_iter)
                    writer.flush()  # Ensure figures are written to disk
                    plt.close(fig)

            # Clean up batch references after visualization
            try:
                del batch, inputs, outputs, targets
            except NameError:
                pass  # Variables may not exist in edge cases

        # Clean up saved training batch variables if they exist
        try:
            del (
                last_train_batch,
                last_train_inputs,
                last_train_outputs,
                last_train_targets,
            )
        except NameError:
            pass  # Variables don't exist if validation ran or training didn't reach the saving step

        # Clear the GPU memory and trigger garbage collection
        _clear_memory(force_gc=True)

    # Close the summarywriter
    writer.close()

    # %%
