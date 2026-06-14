import torch.nn as nn

from .networks.nnformer_tumor import nnFormer


class NNFormer3D(nn.Module):
    """CellMap wrapper around the 3D nnFormer tumor architecture."""

    def __init__(self, n_channels, n_classes, img_size=(128, 128, 128)):
        super().__init__()
        self.net = nnFormer(
            crop_size=list(img_size),
            embedding_dim=96,
            input_channels=n_channels,
            num_classes=n_classes,
            conv_op=nn.Conv3d,
            depths=[2, 2, 2, 2],
            num_heads=[3, 6, 12, 24],
            patch_size=[4, 4, 4],
            window_size=[4, 4, 8, 4],
            deep_supervision=False,
        )

    def forward(self, x):
        outputs = self.net(x)
        if isinstance(outputs, (tuple, list)):
            return outputs[0]
        return outputs
