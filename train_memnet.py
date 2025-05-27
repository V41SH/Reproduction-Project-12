import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from membrane_net import MembraneNet
from your_em_dataset import ISBIDataset  # You need to define this Dataset class


def elastic_net_loss(model, l1=1e-7, l2=1e-8):
    l1_reg = torch.tensor(0., requires_grad=True).to(next(model.parameters()).device)
    l2_reg = torch.tensor(0., requires_grad=True).to(next(model.parameters()).device)
    for name, param in model.named_parameters():
        if 'weight' in name:
            l1_reg = l1_reg + torch.norm(param, 1)
            l2_reg = l2_reg + torch.norm(param, 2)**2
    return l1 * l1_reg + l2 * l2_reg


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        output = output[..., 32:-32, 32:-32]  # Crop back to 256x256
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
        transforms.RandomRotation([0, 90, 180, 270]),
        transforms.ToTensor()
    ])

    train_dataset = ISBIDataset("/path/to/data", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)

    model = MembraneNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.05)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.85)

    epochs = 100
    for epoch in range(epochs):
        loss = train_epoch(model, train_loader, optimizer, device)
        print(f"Epoch {epoch + 1}: Loss = {loss:.4f}")
        scheduler.step()

        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"membrane_net_epoch{epoch+1}.pth")


if __name__ == '__main__':
    main()
