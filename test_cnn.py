import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
import os
from e2cnn import gspaces
from e2cnn import nn as enn

from models.sfcnn import SFCNN
from models.cnn import ComparableCNN

# --------- SETTINGS ---------
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 128
ROTATION_ANGLES = np.arange(0, 360, 15)  # 0 to 345
NUM_CLASSES = 10

# --------- ROTATED TEST SET ---------
class RotatedMNISTTestSet(Dataset):
    def __init__(self, angle):
        self.dataset = datasets.MNIST(root='./data', train=False, download=True)
        self.angle = angle
        self.transform = transforms.Compose([
            transforms.Lambda(lambda x: TF.rotate(x, angle=float(self.angle), interpolation=transforms.InterpolationMode.BILINEAR)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        return self.transform(img), label

def evaluate_model(model, angle):
    testset = RotatedMNISTTestSet(angle)
    loader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            outputs = model(x)
            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    acc = correct / total * 100
    return 100 - acc  # return error %

# --------- MODEL LOADING ---------
def load_models():
    models = {}

    # SFCNN
    sfcnn = SFCNN(num_orientations=16).to(DEVICE)
    sfcnn.load_state_dict(torch.load('/kaggle/input/fig4_right/pytorch/default/1/sfcnn_12k.pth'))
    models['SFCNN (Λ=16)'] = sfcnn

    # CNNs
    for tag in ['none', 'pi_2', 'pi_4', 'continuous']:
        cnn = ComparableCNN().to(DEVICE)
        cnn.load_state_dict(torch.load(f'/kaggle/input/fig4_right/pytorch/default/1/cnn_{tag}.pth'))
        models[f'CNN ({tag})'] = cnn

    return models

# --------- MAIN: EVALUATE ALL ---------
def main():
    models = load_models()
    all_errors = {}

    print("Evaluating models on rotated test sets...")
    for name, model in models.items():
        print(f"→ {name}")
        errors = []
        for angle in ROTATION_ANGLES:
            err = evaluate_model(model, angle)
            errors.append(err)
        all_errors[name] = errors

    # --------- PLOT ---------
    plt.figure(figsize=(10, 6))
    for name, errors in all_errors.items():
        plt.plot(ROTATION_ANGLES, errors, label=name)

    plt.xlabel("Rotation angle (degrees)")
    plt.ylabel("Test Error (%)")
    plt.title("Rotational Generalization")
    plt.grid(True)
    plt.legend()
    plt.xticks(ROTATION_ANGLES)
    plt.tight_layout()
    plt.savefig("figure4_right.png")
    plt.show()
