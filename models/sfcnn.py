import torch
import torch.nn as nn
from e2cnn import gspaces
from e2cnn import nn as enn

class SFCNN(nn.Module):
    def __init__(self, init=True, num_orientations=16):
        super().__init__()

        self.r2_act = gspaces.Rot2dOnR2(N=num_orientations)

        # Input layer: 24 channels, 9x9
        in_type = enn.FieldType(self.r2_act, [self.r2_act.trivial_repr])
        self.input_type = in_type
        out_type = enn.FieldType(self.r2_act, 24*[self.r2_act.regular_repr])
        self.block1 = enn.SequentialModule(
            enn.R2Conv(in_type, out_type, kernel_size=9, padding=4, initialize=init),
            enn.InnerBatchNorm(out_type),
            enn.ReLU(out_type)
        )

        # Layer 2: 32 channels, 7x7
        in_type = out_type
        out_type = enn.FieldType(self.r2_act, 32*[self.r2_act.regular_repr])
        self.block2 = enn.SequentialModule(
            enn.R2Conv(in_type, out_type, kernel_size=7, padding=3, initialize=init),
            enn.InnerBatchNorm(out_type),
            enn.ReLU(out_type)
        )

        # Spatial max pooling 2x2
        self.pool1 = enn.PointwiseMaxPool(out_type, kernel_size=2, stride=2)

        # Layer 3 & 4: 36 channels, 7x7
        in_type = out_type
        out_type = enn.FieldType(self.r2_act, 36*[self.r2_act.regular_repr])
        self.block3 = enn.SequentialModule(
            enn.R2Conv(in_type, out_type, kernel_size=7, padding=3, initialize=init),
            enn.InnerBatchNorm(out_type),
            enn.ReLU(out_type),
            enn.R2Conv(out_type, out_type, kernel_size=7, padding=3, initialize=init),
            enn.InnerBatchNorm(out_type),
            enn.ReLU(out_type)
        )

        # Spatial max pooling 2x2
        self.pool2 = enn.PointwiseMaxPool(out_type, kernel_size=2, stride=2)

        # Layer 5 & 6: 64 and 96 channels
        in_type = out_type
        mid_type = enn.FieldType(self.r2_act, 64*[self.r2_act.regular_repr])
        out_type = enn.FieldType(self.r2_act, 96*[self.r2_act.regular_repr])
        self.block4 = enn.SequentialModule(
            enn.R2Conv(in_type, mid_type, kernel_size=7, padding=3, initialize=init),
            enn.InnerBatchNorm(mid_type),
            enn.ReLU(mid_type),
            enn.R2Conv(mid_type, out_type, kernel_size=5, padding=2, initialize=init),
            enn.InnerBatchNorm(out_type),
            enn.ReLU(out_type)
        )

        # Global spatial pooling
        self.gpool = enn.PointwiseAdaptiveAvgPool(out_type, output_size=1)

        # Global orientation pooling
        self.opool = enn.GroupPooling(out_type)

        # Fully connected layers: 96 -> 96 -> 10
        self.fc = nn.Sequential(
            nn.Linear(96, 96),
            nn.ReLU(),
            nn.Linear(96, 96),
            nn.ReLU(),
            nn.Linear(96, 10)
        )

    def forward(self, x):
        x = enn.GeometricTensor(x, self.input_type)
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool1(x)
        x = self.block3(x)
        x = self.pool2(x)
        x = self.block4(x)
        x = self.gpool(x)
        x = self.opool(x).tensor
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
