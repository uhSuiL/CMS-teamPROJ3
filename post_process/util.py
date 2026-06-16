import os
import numpy as np
import zarr


def load_data(file_domain, crop, class_name, scale='s0') -> np.ndarray:
    file_name = f'{crop}/{class_name}/{scale}'
    file_path = os.path.join(file_domain, file_name)
    data = zarr.open(file_path, mode='r')
    arr = np.array(data, dtype=np.float32)
    return arr


def load_multiclass_data(class_names, file_domain, crop, scale='s0') -> np.ndarray:
    return np.stack([
        load_data(file_domain, crop, cls_name,scale)
        for cls_name in class_names
    ], axis=-1)

