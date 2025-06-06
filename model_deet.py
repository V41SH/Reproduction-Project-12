import torch
from models.memnet import MembraneNet

state_dict = torch.load("membrane_net_epoch40.pth", map_location="cpu")

for k, v in state_dict.items():
    print(f"{k}: {tuple(v.shape)}")
