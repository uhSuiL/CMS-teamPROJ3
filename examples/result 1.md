result 1

```python
learning_rate = 0.0002  # learning rate for the optimizer
batch_size = 1  # batch size for the dataloader
input_array_info = {
    "shape": (128, 128, 128),
    "scale": (8, 8, 8),
}  # shape and voxel size of the data to load for the input
target_array_info = {
    "shape": (128, 128, 128),
    "scale": (8, 8, 8),
}  # shape and voxel size of the data to load for the target
epochs = 120  # number of epochs to train the model for
iterations_per_epoch = 3  # number of iterations per epoch
random_seed = 42  # random seed for reproducibility


model_name = "3d_transunet"  # name of the model to use
model_to_load = "3d_transunet"  # name of the pre-trained model to load
model = TransUNet_3D(1, len(classes), img_size=input_array_info["shape"])

classes = ["endo_lum", "cyto", "endo_mem", "bg"]
force_all_classes = True

# Training loss: patch-wise normalized inverse-frequency weighted CE.
criterion = CellMapDynamicWeightedCrossEntropyLoss
criterion_kwargs = {"max_class_weight": 25.0}
wrap_loss = False
weight_loss = False
```



endo_lum: min=-1.6632, max= 1.8938, mean=-0.2920, argmax=95,697 (1.20%)
    cyto: min=-2.2941, max= 1.3344, mean=-0.3218, argmax=12,232 (0.15%)
endo_mem: min=-1.5364, max= 1.0369, mean=-0.5307, argmax=4,603 (0.06%)
      bg: min=-0.6218, max= 2.6642, mean= 1.6065, argmax=7,887,468 (98.59%)



GT endo_lum: 119,012 (1.4876%)
GT     cyto: 3,960,320 (49.5040%)
GT endo_mem: 116,085 (1.4511%)
GT       bg: 3,804,583 (47.5573%)



![image-20260624060121105](C:\Users\yanch\AppData\Roaming\Typora\typora-user-images\image-20260624060121105.png)

![image-20260624060150408](C:\Users\yanch\AppData\Roaming\Typora\typora-user-images\image-20260624060150408.png)

![check_crop124_logits](E:\teamproject\code\CMS-teamPROJ3\examples\check_crop124_logits.png)

失败 唯一欣慰的是基本的类别能多少logits内部能识别出来





patch测试：

![image-20260624062147656](C:\Users\yanch\AppData\Roaming\Typora\typora-user-images\image-20260624062147656.png)

很多分块里没有cyto!!!

![image-20260624063236389](C:\Users\yanch\AppData\Roaming\Typora\typora-user-images\image-20260624063236389.png)

![image-20260624063254334](C:\Users\yanch\AppData\Roaming\Typora\typora-user-images\image-20260624063254334.png)

 result 2

endo_lum: min=-2.3006, max= 0.4108, mean=-0.9282, argmax=31 (0.00%)
    cyto: min=-1.7968, max= 0.9835, mean=-0.3920, argmax=166,846 (2.09%)
endo_mem: min=-1.1294, max= 1.1379, mean=-0.5056, argmax=2,805 (0.04%)
      pm: min=-1.2600, max= 1.8651, mean= 0.1457, argmax=1,719,276 (21.49%)
     ecs: min=-1.7725, max= 0.7708, mean=-0.2311, argmax=789,248 (9.87%)
      bg: min=-1.0331, max= 0.7994, mean= 0.2550, argmax=5,321,794 (66.52%)

Groundtruth path: E:\teamproject\code\CMS-teamPROJ3\data\jrc_mus-liver\jrc_mus-liver.zarr\recon-1\labels\groundtruth\crop124
Groundtruth shape: (6, 200, 200, 200) = [classes, z, y, x]
Unlabeled voxels among these 6 classes: 0 / 8,000,000
Overlapping voxels among these 6 classes: 0 / 8,000,000

GT endo_lum: 119,012 (1.4876%)
GT     cyto: 3,960,320 (49.5040%)
GT endo_mem: 116,085 (1.4511%)
GT       pm: 535,177 (6.6897%)
GT      ecs: 1,684,813 (21.0602%)
GT       bg: 1,584,593 (19.8074%)

<img src="E:\teamproject\code\CMS-teamPROJ3\examples\check_crop124_logits.png" alt="check_crop124_logits" style="zoom:200%;" />

![check_crop124_overlay](E:\teamproject\code\CMS-teamPROJ3\examples\check_crop124_overlay.png)

![check_crop124_gt_overlay](E:\teamproject\code\CMS-teamPROJ3\examples\check_crop124_gt_overlay.png)

```python
# # 3D TransUNet
model_name = "3d_transunet_6class"
model_to_load = "3d_transunet_6class"
model = TransUNet_3D(1, len(classes), img_size=input_array_info["shape"])

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
epochs = 120  # number of epochs to train the model for
iterations_per_epoch = 3  # number of iterations per epoch
random_seed = 42  # random seed for reproducibility

# classes = ["nuc", "er"]  # list of classes to segment
# classes = get_tested_classes()  # list of classes to segment
classes = ["endo_lum", "cyto", "endo_mem", "pm", "ecs", "bg"]
force_all_classes = True
# These six labels are a custom mutually exclusive label set. Save every model
# channel for explicitly requested numeric crops instead of consulting the
# official challenge test-label manifest, which does not know about custom bg.
predict_filter_classes = False

# Training loss: patch-wise normalized inverse-frequency weighted CE.
criterion = CellMapDynamicWeightedCrossEntropyLoss
criterion_kwargs = {
    "min_class_weight": 0.001,
    "max_class_weight": 100.0,
}
wrap_loss = False
weight_loss = False

# Validation always measures ordinary, unweighted cross-entropy.
validation_criterion = CellMapCrossEntropyLoss
validation_criterion_kwargs = {}
validation_wrap_loss = False

# # Defining model (comment out all that are not used)
# # 3D UNet
model_name = "3d_unet_6class"  # keep six-class checkpoints separate
model_to_load = "3d_unet_6class"
model = UNet_3D(1, len(classes))
```

result 3

```python
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
epochs = 120  # number of epochs to train the model for
iterations_per_epoch = 3  # number of iterations per epoch
random_seed = 42  # random seed for reproducibility

# classes = ["nuc", "er"]  # list of classes to segment
# classes = get_tested_classes()  # list of classes to segment
classes = ["endo_lum", "cyto", "endo_mem", "pm", "ecs", "bg"]
force_all_classes = True
# These six labels are a custom mutually exclusive label set. Save every model
# channel for explicitly requested numeric crops instead of consulting the
# official challenge test-label manifest, which does not know about custom bg.
predict_filter_classes = False

# Training loss: patch-wise normalized inverse-frequency weighted CE.
criterion = CellMapDynamicWeightedCrossEntropyLoss
criterion_kwargs = {
    "min_class_weight": 0.001,
    "max_class_weight": 100.0,
}
wrap_loss = False
weight_loss = False

# Validation always measures ordinary, unweighted cross-entropy.
validation_criterion = CellMapCrossEntropyLoss
validation_criterion_kwargs = {}
validation_wrap_loss = False

# # Defining model (comment out all that are not used)
# # 3D UNet
model_name = "3d_unet_6class"  # keep six-class checkpoints separate
model_to_load = "3d_unet_6class"
model = UNet_3D(1, len(classes))
```

endo_lum: min=-0.5105, max= 0.5711, mean= 0.0601, argmax=3,606 (0.05%)
    cyto: min=-0.6381, max= 0.6202, mean=-0.2367, argmax=2,439 (0.03%)
endo_mem: min=-0.3825, max= 0.4668, mean=-0.0247, argmax=1,525 (0.02%)
      pm: min=-0.5952, max= 0.7206, mean=-0.0544, argmax=5,848 (0.07%)
     ecs: min=-0.7452, max= 0.8538, mean= 0.1432, argmax=392,563 (4.91%)
      bg: min=-0.1719, max= 0.9132, mean= 0.5688, argmax=7,594,019 (94.93%)

![check_crop124_logits](E:\teamproject\code\CMS-teamPROJ3\examples\check_crop124_logits.png)

![check_crop124_overlay](E:\teamproject\code\CMS-teamPROJ3\examples\check_crop124_overlay.png)

result 4



```python
epochs = 120  # number of epochs to train the model for
iterations_per_epoch = 3  # number of iterations per epoch
random_seed = 42  # random seed for reproducibility

# classes = ["nuc", "er"]  # list of classes to segment
# classes = get_tested_classes()  # list of classes to segment
classes = ["endo_lum", "cyto", "endo_mem", "pm", "ecs", "bg"]
force_all_classes = True
# These six labels are a custom mutually exclusive label set. Save every model
# channel for explicitly requested numeric crops instead of consulting the
# official challenge test-label manifest, which does not know about custom bg.
predict_filter_classes = False

# Training loss: equally weighted ordinary multi-class CE and soft Dice.
criterion = CellMapDiceCELoss
criterion_kwargs = {
    "ce_weight": 0.5,
    "dice_weight": 0.5,
}
model_name = "3d_transunet_6class"
model_to_load = "3d_transunet_6class"
model = TransUNet_3D(1, len(classes), img_size=input_array_info["shape"])
```

endo_lum: min=-2.0046, max= 1.1163, mean=-0.5533, argmax=84 (0.00%)
    cyto: min=-1.7988, max= 2.0479, mean= 0.6745, argmax=502,530 (6.28%)
endo_mem: min=-1.9646, max= 0.3647, mean=-0.9409, argmax=0 (0.00%)
      pm: min=-1.2058, max= 1.9001, mean=-0.2489, argmax=3,874 (0.05%)
     ecs: min=-2.3486, max= 2.3798, mean= 1.1220, argmax=2,266,101 (28.33%)
      bg: min=-0.4973, max= 2.8836, mean= 1.3543, argmax=5,227,411 (65.34%)

GT endo_lum: 119,012 (1.4876%)
GT     cyto: 3,960,320 (49.5040%)
GT endo_mem: 116,085 (1.4511%)
GT       pm: 535,177 (6.6897%)
GT      ecs: 1,684,813 (21.0602%)
GT       bg: 1,584,593 (19.8074%)

![check_crop124_logits](E:\teamproject\code\CMS-teamPROJ3\examples\check_crop124_logits.png)



![check_crop124_overlay](E:\teamproject\code\CMS-teamPROJ3\examples\check_crop124_overlay.png)

![check_crop124_gt_overlay](E:\teamproject\code\CMS-teamPROJ3\examples\check_crop124_gt_overlay.png)

结论：排除模型问题！而且我个人认为是bg太杂了，导致模型认为什么都是bg类别





CE+高置信规则

![image-20260624232839824](C:\Users\yanch\AppData\Roaming\Typora\typora-user-images\image-20260624232839824.png)

CE+bg惩罚结果

![check_crop124_logits](E:\teamproject\code\CMS-teamPROJ3\examples\check_crop124_logits.png)

![check_crop124_overlay](E:\teamproject\code\CMS-teamPROJ3\examples\check_crop124_overlay.png)

endo_lum: min=-2.3175, max= 0.6662, mean=-1.1697, argmax=0 (0.00%)
    cyto: min=-6.1386, max= 2.7578, mean= 0.9274, argmax=296,205 (3.70%)
endo_mem: min=-1.2218, max= 1.9419, mean= 0.2581, argmax=1 (0.00%)
      pm: min=-1.0866, max= 3.1474, mean= 0.3222, argmax=6,573 (0.08%)
     ecs: min=-1.4158, max= 4.5661, mean= 2.7076, argmax=7,212,999 (90.16%)
      bg: min= 0.2114, max= 3.8559, mean= 2.3308, argmax=484,222 (6.05%)