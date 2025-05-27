import torch
import torch.nn as nn
from e2cnn import gspaces
from e2cnn import nn as enn

class ResidualBlock(nn.Module):
    def __init__(self, r2_act, in_type):
        super().__init__()
        self.out_type = in_type
        self.block = enn.SequentialModule(
            enn.R2Conv(in_type, in_type, kernel_size=7, padding=3),
            enn.InnerBatchNorm(in_type),
            enn.ReLU(in_type)
        )

    def forward(self, x):
        return x + self.block(x)

class MembraneNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.r2_act = gspaces.Rot2dOnR2(N=17)

        # Encoder
        def enc_block(in_repr, out_channels):
            out_repr = enn.FieldType(self.r2_act, out_channels * [self.r2_act.regular_repr])
            block = enn.SequentialModule(
                enn.R2Conv(in_repr, out_repr, kernel_size=7, padding=3),
                enn.InnerBatchNorm(out_repr),
                enn.ReLU(out_repr),
                ResidualBlock(self.r2_act, out_repr)
            )
            return block, out_repr

        # Input
        in_type = enn.FieldType(self.r2_act, [self.r2_act.trivial_repr])
        feat_12 = enn.FieldType(self.r2_act, 12 * [self.r2_act.regular_repr])
        self.input_layer = enn.SequentialModule(
            enn.R2Conv(in_type, feat_12, kernel_size=11, padding=5),
            enn.InnerBatchNorm(feat_12),
            enn.ReLU(feat_12)
        )

        # Down blocks
        self.enc1, feat_24 = enc_block(feat_12, 24)
        self.pool1 = enn.PointwiseMaxPool(feat_24, 2, 2)
        self.enc2, feat_48 = enc_block(feat_24, 48)
        self.pool2 = enn.PointwiseMaxPool(feat_48, 2, 2)
        self.enc3, feat_96 = enc_block(feat_48, 96)

        # Decoder
        self.up1 = enn.PointwiseUpsampling(feat_96, 2)
        self.dec1, _ = enc_block(feat_96, 48)
        self.up2 = enn.PointwiseUpsampling(feat_48, 2)
        self.dec2, _ = enc_block(feat_48, 24)
        self.up3 = enn.PointwiseUpsampling(feat_24, 2)
        self.dec3, _ = enc_block(feat_24, 12)

        self.orientation_pool = enn.GroupPooling(feat_12)

        # Final 1x1 convs
        self.final = nn.Sequential(
            nn.Conv2d(12, 12, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(12, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = enn.GeometricTensor(x, enn.FieldType(self.r2_act, [self.r2_act.trivial_repr]))
        x0 = self.input_layer(x)

        x1 = self.enc1(x0)
        x2 = self.enc2(self.pool1(x1))
        x3 = self.enc3(self.pool2(x2))

        d1 = self.dec1(self.up1(x3))
        d1 = d1 + x2
        d2 = self.dec2(self.up2(d1))
        d2 = d2 + x1
        d3 = self.dec3(self.up3(d2))
        d3 = d3 + x0

        y = self.orientation_pool(d3).tensor
        return self.final(y)
