predicted data:

csc predict train_3D.py -c 252,235,117,134,174,1,14,155,6,8,27,34,61,85,87,180,47,92,93,109,72,148,160,164,277,280,320,369,125,133,144,150,115,98,188,189,202,192,228 -O -s "E:\teamproject\code\CMS-teamPROJ3\data\{dataset}\{dataset}.zarr\recon-1\{name}" -r "em\fibsem-uint8"

training args:

```python
# %% Set hyperparameters and other configurations
learning_rate = 0.0001  # learning rate for the optimizer
batch_size = 16  # final valid batch size after patch filtering
input_array_info = {
    "shape": (128, 128, 128),
    "scale": (8, 8, 8),
}  # shape and voxel size of the data to load for the input
target_array_info = {
    "shape": (128, 128, 128),
    "scale": (8, 8, 8),
}  # shape and voxel size of the data to load for the target
epochs = 200  # number of epochs to train the model for
iterations_per_epoch = 15  # 150 epochs * 120 iterations ~= a long supercomputer run
random_seed = 42  # random seed for reproducibility

# classes = ["nuc", "er"]  # list of classes to segment
# classes = get_tested_classes()  # list of classes to segment
classes = ["endo_lum", "cyto", "endo_mem", "pm", "ecs", "bg"]
target_classes = classes
force_all_classes = True
# Save all six model classes for explicitly requested numeric crops.
predict_filter_classes = False

# Formal six-class experiment:
# 0.4 dynamic inverse-frequency CE + 0.6 Dice.
# Patches where ecs + bg occupy more than 60% are rejected before model forward.
patch_filter_class_indices = [4, 5]  # ["ecs", "bg"]
patch_filter_ratio_threshold = 0.65
patch_filter_max_attempts = 200
patch_filter_min_batch_size = 16

criterion = CellMapFilteredDynamicWeightedDiceCELoss
criterion_kwargs = {
    "ce_weight": 0.5,
    "dice_weight": 0.5,
    "dice_smooth": 1,
    "min_class_weight": 0.1,
    "max_class_weight": 10.0,
    "include_background": True,
}
wrap_loss = False
weight_loss = False

# Validation uses plain multi-class CE as a simple unweighted metric.
validation_criterion = CellMapCrossEntropyLoss
validation_criterion_kwargs = {}
validation_wrap_loss = False

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
```

