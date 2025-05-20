import torch
import torch.nn as nn
from e2cnn import gspaces
from e2cnn import nn as enn

class SFCNN(nn.Module):
    def __init__(self, num_orientations=8):
        super().__init__()
        
        # The symmetry group is N rotations (discrete cyclic group)
        self.r2_act = gspaces.Rot2dOnR2(N=num_orientations)
        
        # Input layer
        in_type = enn.FieldType(self.r2_act, [self.r2_act.trivial_repr])
        self.input_type = in_type
        
        # Layer 1: 24 channels, 9x9 kernel
        out_type = enn.FieldType(self.r2_act, 24*[self.r2_act.regular_repr])
        self.block1 = enn.SequentialModule(
            enn.R2Conv(in_type, out_type, kernel_size=9, padding=4),
            enn.InnerBatchNorm(out_type),
            enn.ReLU(out_type),
            enn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
        )
        
        # Layer 2: 32 channels, 7x7 kernel
        in_type = out_type
        out_type = enn.FieldType(self.r2_act, 32*[self.r2_act.regular_repr])
        self.block2 = enn.SequentialModule(
            enn.R2Conv(in_type, out_type, kernel_size=7, padding=3),
            enn.InnerBatchNorm(out_type),
            enn.ReLU(out_type),
            enn.PointwiseAvgPoolAntialiased(out_type, sigma=0.66, stride=2)
        )
        
        # Layer 3: 64 channels, 5x5 kernel
        in_type = out_type
        out_type = enn.FieldType(self.r2_act, 64*[self.r2_act.regular_repr])
        self.block3 = enn.SequentialModule(
            enn.R2Conv(in_type, out_type, kernel_size=5, padding=2),
            enn.InnerBatchNorm(out_type),
            enn.ReLU(out_type)
        )
        
        # Final pooling and FC layers
        self.pool_final = enn.GroupPooling(out_type)
        self.fc = nn.Sequential(
            nn.Linear(64*7*7, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        # Convert input to geometric tensor
        x = enn.GeometricTensor(x, self.input_type)
        
        # Forward through blocks
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        # Final pooling and classification
        x = self.pool_final(x).tensor
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x