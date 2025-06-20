import torch
import torch.nn as nn
import torch.nn.functional as F
from e2cnn import gspaces
from e2cnn import nn as enn

# Residual block of depth 1 used in the paper
class ResidualBlock(enn.EquivariantModule):
    def __init__(self, r2_act, in_type):
        super().__init__()
        self.in_type = in_type
        self.out_type = in_type
        self.block = enn.SequentialModule(
            enn.R2Conv(in_type, in_type, kernel_size=7, padding=3),
            enn.InnerBatchNorm(in_type),
            enn.ReLU(in_type)
        )

    def forward(self, x):
        return x + self.block(x)

    def evaluate_output_type(self, input_type):
        return self.out_type

    def evaluate_output_shape(self, input_shape):
        return input_shape


# Upsamples the input
class PointwiseUpsample(enn.EquivariantModule):
    def __init__(self, in_type, scale_factor):
        super().__init__()
        self.in_type = in_type
        self.out_type = in_type
        self.scale_factor = scale_factor

    def forward(self, x):
        x.tensor = F.interpolate(x.tensor, scale_factor=self.scale_factor, mode='nearest')
        return x

    def evaluate_output_type(self, input_type):
        return self.out_type

    def evaluate_output_shape(self, input_shape):
        N, C, H, W = input_shape
        return N, C, H * self.scale_factor, W * self.scale_factor

# U-net based architecture for ISBI EM Segmentation Challenge
class MembraneNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.r2_act = gspaces.Rot2dOnR2(N=17)  # 17 orientations
        self.dropout_prob = 0.4  # As specified in paper for ISBI experiment

        # Encoder blocks with dropout
        def enc_block(in_repr, out_channels):
            out_repr = enn.FieldType(self.r2_act, out_channels * [self.r2_act.regular_repr])
            block = enn.SequentialModule(
                enn.R2Conv(in_repr, out_repr, kernel_size=7, padding=3),
                enn.InnerBatchNorm(out_repr),
                enn.ReLU(out_repr),
                enn.FieldDropout(out_repr, p=self.dropout_prob),  # Dropout after each block
                ResidualBlock(self.r2_act, out_repr)
            )
            return block, out_repr

        # Input layer with dropout
        in_type = enn.FieldType(self.r2_act, [self.r2_act.trivial_repr])
        feat_12 = enn.FieldType(self.r2_act, 12 * [self.r2_act.regular_repr])
        self.input_layer = enn.SequentialModule(
            enn.R2Conv(in_type, feat_12, kernel_size=11, padding=5),
            enn.InnerBatchNorm(feat_12),
            enn.ReLU(feat_12),
            enn.FieldDropout(feat_12, p=self.dropout_prob)  # Dropout after input
        )

        # Down blocks - each will have dropout from enc_block
        self.enc1, feat_24 = enc_block(feat_12, 24)
        self.pool1 = enn.PointwiseMaxPool(feat_24, 2, 2)
        self.enc2, feat_48 = enc_block(feat_24, 48)
        self.pool2 = enn.PointwiseMaxPool(feat_48, 2, 2)
        self.enc3, feat_48_2 = enc_block(feat_48, 48)
        self.pool3 = enn.PointwiseMaxPool(feat_48_2, 2, 2)
        self.enc4, feat_48_3 = enc_block(feat_48_2, 48)
        self.pool4 = enn.PointwiseMaxPool(feat_48_3, 2, 2)
        self.enc5, feat_48_4 = enc_block(feat_48_3, 48)

        # Decoder blocks - add dropout to each decoder block too
        def dec_block(in_repr, out_channels):
            out_repr = enn.FieldType(self.r2_act, out_channels * [self.r2_act.regular_repr])
            return enn.SequentialModule(
                enn.R2Conv(in_repr, out_repr, kernel_size=7, padding=3),
                enn.InnerBatchNorm(out_repr),
                enn.ReLU(out_repr),
                enn.FieldDropout(out_repr, p=self.dropout_prob)  # Dropout in decoder
            ), out_repr

        self.up1 = PointwiseUpsample(feat_48_4, 2)
        self.dec1, _ = dec_block(feat_48_4, 48)
        self.up2 = PointwiseUpsample(feat_48_3, 2)
        self.dec2, _ = dec_block(feat_48_3, 48)
        self.up3 = PointwiseUpsample(feat_48_2, 2)
        self.dec3, _ = dec_block(feat_48_2, 24)
        self.up4 = PointwiseUpsample(feat_24, 2)
        self.dec4, _ = dec_block(feat_24, 12)

        self.orientation_pool = enn.GroupPooling(feat_12)

        # Final 1x1 convs with regular dropout
        self.final = nn.Sequential(
            nn.Conv2d(12, 12, kernel_size=1),
            nn.ReLU(),
            nn.Dropout2d(p=self.dropout_prob),  # Regular dropout for conventional layers
            nn.Conv2d(12, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = enn.GeometricTensor(x, enn.FieldType(self.r2_act, [self.r2_act.trivial_repr]))
        x0 = self.input_layer(x)

        x1 = self.enc1(x0)
        x2 = self.enc2(self.pool1(x1))
        x3 = self.enc3(self.pool2(x2))
        x4 = self.enc4(self.pool3(x3))
        x5 = self.enc5(self.pool4(x4))

        d1 = self.up1(x5)
        d1 = d1 + x4
        d1 = self.dec1(d1)
        d2 = self.up2(d1)
        d2 = d2 + x3
        d2 = self.dec2(d2)
        d3 = self.up3(d2)
        d3 = d3 + x2
        d3 = self.dec3(d3)
        d4 = self.up4(d3)
        d4 = d4 + x1
        d4 = self.dec4(d4)

        y = self.orientation_pool(d4).tensor
        return self.final(y)