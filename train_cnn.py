import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import os
import math
from datetime import datetime
from e2cnn import nn as enn
from torchvision import datasets, transforms

from mnist_rot.data_loader_mnist_rot import build_mnist_rot_loader
from models.sfcnn import SFCNN

from models.cnn import ComparableCNN

# ------------------ AUGMENTATION SCHEMES ------------------

def get_mnist_train_loader(batch_size, rotation_aug='none'):
    angle_map = {
        'none': [],
        'pi_2': [0, 90, 180, 270],
        'pi_4': [i * 45 for i in range(8)],
        'continuous': list(range(360))
    }

    def random_rotate(img):
        angle = torch.tensor(angle_map[rotation_aug])[torch.randint(0, len(angle_map[rotation_aug]), (1,))].item() \
            if rotation_aug in ['pi_2', 'pi_4'] else torch.rand(1).item() * 360
        return transforms.functional.rotate(img, angle, interpolation=transforms.InterpolationMode.BILINEAR)

    transform = transforms.Compose([
        transforms.Lambda(random_rotate) if rotation_aug != 'none' else transforms.Lambda(lambda x: x),
        transforms.ToTensor()
    ])

    dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)


def get_mnist_val_loader(batch_size):
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

def apply_standard_he_init(model):
    for m in model.modules():
        if isinstance(m, enn.R2Conv):
            # m.weights is a 1D tensor of shape (num_basis,)
            std = math.sqrt(2.0 / m.in_type.size)  # fan_in = number of input fields
            with torch.no_grad():
                m.weights.data.normal_(0, std)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for inputs, targets in tqdm(loader, desc="Training"):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return total_loss / len(loader), 100. * correct / total

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Validating"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return total_loss / len(loader), 100. * correct / total

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['cnn', 'sfcnn'], required=True)
    parser.add_argument('--augmentation', type=str, default='none', choices=['none', 'pi_2', 'pi_4', 'continuous'])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--lr', type=float, default=0.015)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    exp_dir = f"outputs/{timestamp}_{args.model}_{args.augmentation}"
    os.makedirs(exp_dir, exist_ok=True)

    print(f"Training {args.model.upper()} with augmentation: {args.augmentation}")

    # Data
    train_loader = get_mnist_train_loader(args.batch_size, rotation_aug=args.augmentation)
    val_loader = get_mnist_val_loader(args.batch_size)

    # Model
    if args.model == 'cnn':
        model = ComparableCNN().to(args.device)
    else:
        model = SFCNN(num_orientations=16).to(args.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda e: 1.0 if e < 15 else 0.8 ** (e - 14))

    # Training Loop
    best_val_acc = 0
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, args.device)
        val_loss, val_acc = validate(model, val_loader, criterion, args.device)
        scheduler.step()

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{exp_dir}/best_model.pth")
            print("Saved new best model!")

    print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.2f}%")

if __name__ == "__main__":
    main()
