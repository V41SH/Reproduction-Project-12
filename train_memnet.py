import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import random
from torchvision.transforms import functional as TF

from models.memnet import MembraneNet
from ISBI.dataloader import ISBIDataset

def elastic_net_loss(model, l1=1e-7, l2=1e-8):
    l1_reg = torch.tensor(0., requires_grad=True).to(next(model.parameters()).device)
    l2_reg = torch.tensor(0., requires_grad=True).to(next(model.parameters()).device)
    for name, param in model.named_parameters():
        if 'weight' in name:
            l1_reg = l1_reg + torch.norm(param, 1)
            l2_reg = l2_reg + torch.norm(param, 2)**2
    return l1 * l1_reg + l2 * l2_reg

class RandomRotation90:
    def __call__(self, x):
        angle = random.choice([0, 90, 180, 270])
        return TF.rotate(x, angle)

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        output = output[..., 16:-16, 16:-16]
        target = target[..., 16:-16, 16:-16]
        loss = F.binary_cross_entropy(output, target) + elastic_net_loss(model)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        RandomRotation90(),
        transforms.ToTensor()
    ])

    train_dataset = ISBIDataset("ISBI/train-volume.tif", "ISBI/train-labels.tif", patch_size=256, pad=32, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)

    model = MembraneNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.05)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.85)

    epochs = 100
    for epoch in tqdm(range(epochs)):
    # for epoch in range(epochs):
        print(f"\nStarting epoch {epoch + 1}")
        loss = train_epoch(model, train_loader, optimizer, device)
        print(f"Epoch {epoch + 1}: Loss = {loss:.4f}")

        print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e6:.1f} MB")
        print(f"Memory reserved: {torch.cuda.memory_reserved() / 1e6:.1f} MB")

        scheduler.step()

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"membrane_net_epoch{epoch+1}.pth")


if __name__ == '__main__':
    main()
