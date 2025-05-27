import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import os
import math
from datetime import datetime
from e2cnn import nn as enn

from mnist_rot.data_loader_mnist_rot import build_mnist_rot_loader
from models.sfcnn import SFCNN
import torch.nn.init as init

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
    parser.add_argument('--num_orientations', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--lr', type=float, default=0.015)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'mps')
    parser.add_argument('--init', type=str, default='coeff')
    args = parser.parse_args()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs(f"outputs/{timestamp}", exist_ok=True)

    train_loader, _, _ = build_mnist_rot_loader('train', args.batch_size)
    val_loader, _, _ = build_mnist_rot_loader('valid', args.batch_size)

    if args.init == 'coeff':
        model = SFCNN(args.num_orientations).to(args.device)
    elif args.init == 'he':
        model = SFCNN(num_orientations=args.num_orientations).to(args.device)
        apply_standard_he_init(model)
    elif args.init == 'no':
        model = SFCNN(init=False, num_orientations=args.num_orientations).to(args.device)

    criterion = nn.CrossEntropyLoss()

    # Split conv and fc params for different weight decays
    conv_params = []
    fc_params = []
    for name, param in model.named_parameters():
        if 'fc' in name:
            fc_params.append(param)
        else:
            conv_params.append(param)

    optimizer = optim.Adam([
        {'params': conv_params, 'weight_decay': 1e-7},
        {'params': fc_params, 'weight_decay': 1e-8}
    ], lr=args.lr)

    # LR scheduler: decay by 0.8 per epoch from epoch 15
    def lr_lambda(epoch):
        return 1.0 if epoch < 15 else 0.8 ** (epoch - 14)

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

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
            torch.save(model.state_dict(), f"outputs/{timestamp}/{args.num_orientations}best_model.pth")
            print("Saved new best model!")

    print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.2f}%")

if __name__ == "__main__":
    main()
