import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dict_of_encoders = {
    "se50": [2048, 1024, 512, 1024, "resnext50_32x4d"],
    "res50": [2048, 1024, 512, 1024, "resnet50"],
    "res34": [512, 256, 128, 512, "resnet50"],
    "d169": [1664, 1280, 512, 768, "densenet169"],
    "b1": [320, 112, 40, 384, "efficientnet-b1"],
    "b2": [352, 120, 48, 384, "efficientnet-b2"],
    "b3": [384, 136, 48, 512, "efficientnet-b3"],
    "b4": [448, 160, 56, 1024, "efficientnet-b4"],
}


class double_conv(nn.Module):
    """(conv => BN => ReLU) * 2"""

    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2=None):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))

        if x2 is not None:
            x = torch.cat([x2, x1], dim=1)
        else:
            x = x1

        x = self.conv(x)

        return x


def get_mesh(batch_size, shape_x, shape_y):
    mg_x, mg_y = np.meshgrid(np.linspace(0, 1, shape_y), np.linspace(0, 1, shape_x))
    mg_x = np.tile(mg_x[None, None, :, :], [batch_size, 1, 1, 1]).astype("float32")
    mg_y = np.tile(mg_y[None, None, :, :], [batch_size, 1, 1, 1]).astype("float32")
    mesh = torch.cat([torch.tensor(mg_x).to(device), torch.tensor(mg_y).to(device)], 1)
    return mesh


class Model(nn.Module):
    def __init__(self, enc_name):
        super().__init__()

        size1, size2, size3, z, name = dict_of_encoders[enc_name]

        self.backbone = smp.Unet(
            name, classes=3, encoder_weights="imagenet", activation="sigmoid"
        )

        self.up1 = up(size1 + size2, z)
        self.up2 = up(z + size3, 256)

        self.outc = nn.Conv2d(256 + 2, 8, 1)

    def forward(self, x):

        b_size = x.shape[0]

        enc = self.backbone.encoder(x)

        features = enc[1:]
        features = features[::-1]

        head = features[0]
        skips = features[1:]

        x = self.up1(head, skips[0])
        x = self.up2(x, skips[1])

        mesh_grid = get_mesh(b_size, x.shape[2], x.shape[3])
        x = torch.cat([x, mesh_grid], 1)

        x = self.outc(x)

        return x
