# This is an example of a training configuration file that trains a 3D U-Net model to predict nuclei and endoplasmic reticulum in the CellMap Segmentation Challenge dataset.

# The configuration file defines the hyperparameters, model, and other configurations required for training the model. The `train` function is then called with the configuration file as an argument to start the training process. The `train` function reads the configuration file, sets up the data loaders, model, optimizer, loss function, and other components, and trains the model for the specified number of epochs.

# The configuration file includes the following components:
# 1. Hyperparameters: learning rate, batch size, input and target array information, epochs, iterations per epoch, random seed, and initial number of features for the model.
# 2. Model: 3D U-Net model with two classes (nuclei and endoplasmic reticulum). (You can also use a 3D ResNet or 3D ViT VNet model by uncommenting the relevant lines.)
# 3. Paths: paths for saving logs, model checkpoints, and data split file.
# 4. Spatial transformations: spatial transformations to apply to the training data.

# This configuration file can be used to run training via two different commands:
# 1. `python train_3D.py`: Run the training script directly.
# 2. `csc train train_3D.py`: Run the training script using the `csc train` command-line interface.

# Training progress can be monitored using TensorBoard by running `tensorboard --logdir tensorboard` in the terminal.

# Once the model is trained, you can use the `predict` function to make predictions on new data using the trained model. See the `predict_3D.py` example for more details.

# %%
import torch
from upath import UPath
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from cellmap_segmentation_challenge.models import (
    NNFormer3D,
    ResNet,
    SegFormer3D,
    TransUNet_3D,
    UNet_3D,
    ViTVNet,
)
from cellmap_segmentation_challenge.utils import get_tested_classes

# %% Set hyperparameters and other configurations
learning_rate = 0.0002  # learning rate for the optimizer
batch_size = 2  # batch size for the dataloader
input_array_info = {
    "shape": (128, 128, 128),
    "scale": (8, 8, 8),
}  # shape and voxel size of the data to load for the input
target_array_info = {
    "shape": (128, 128, 128),
    "scale": (8, 8, 8),
}  # shape and voxel size of the data to load for the target
epochs = 150  # number of epochs to train the model for
iterations_per_epoch = 3  # number of iterations per epoch
random_seed = 42  # random seed for reproducibility

# classes = ["nuc", "er"]  # list of classes to segment
# classes = get_tested_classes()  # list of classes to segment
classes = ["endo_lum", "cyto", "endo_mem", "pm", "ecs"]
target_classes = classes
force_all_classes = True
# Save all five model classes for explicitly requested numeric crops.
predict_filter_classes = False

# Explicit BCE configuration.
# Training uses the original CellMap behavior: class-frequency pos_weight is
# added automatically, then CellMapLossWrapper ignores NaN target voxels.
criterion = torch.nn.BCEWithLogitsLoss
criterion_kwargs = {}
wrap_loss = True
weight_loss = True

# Validation also uses BCE, without training-set class weighting.
validation_criterion = torch.nn.BCEWithLogitsLoss
validation_criterion_kwargs = {}
validation_wrap_loss = True

# # Defining model (comment out all that are not used)
# # 3D UNet
# model_name = "3d_unet_6class"  # keep six-class checkpoints separate
# model_to_load = "3d_unet_6class"
# model = UNet_3D(1, len(classes))

# 3D ResNet
# model_name = "3d_resnet_6class"
# model_to_load = "3d_resnet_6class"
# model = ResNet(ndims=3, output_nc=len(classes))

# # 3D TransUNet
model_name = "3d_transunet_6class"
model_to_load = "3d_transunet_6class"
model = TransUNet_3D(1, len(classes), img_size=input_array_info["shape"])

# 3D SegFormer
# model_name = "3d_segformer_6class"
# model_to_load = "3d_segformer_6class"
# model = SegFormer3D(in_channels=1, num_classes=len(classes))

# 3D nnFormer
# model_name = "3d_nnformer_6class"
# model_to_load = "3d_nnformer_6class"
# model = NNFormer3D(1, len(classes), img_size=input_array_info["shape"])

load_model = "latest"  # load the latest model or the best validation model

# Define the paths for saving the model and logs, etc.
logs_save_path = UPath(
    "tensorboard/{model_name}"
).path  # path to save the logs from tensorboard
model_save_path = UPath(
    "checkpoints/{model_name}_{epoch}.pth"  # path to save the model checkpoints
).path
datasplit_path = "datasplit_6class.csv"

# Define the spatial transformations to apply to the training data
spatial_transforms = {  # dictionary of spatial transformations to apply to the data
    "mirror": {"axes": {"x": 0.5, "y": 0.5, "z": 0.1}},
    "transpose": {"axes": ["x", "y", "z"]},
    "rotate": {"axes": {"x": [-180, 180], "y": [-180, 180], "z": [-180, 180]}},
}

# Set a limit to how long the validation can take
validation_time_limit = 25  # time limit in seconds for the validation step
filter_by_scale = True  # filter the data by scale

if __name__ == "__main__":
    from cellmap_segmentation_challenge import train

    train(__file__)
