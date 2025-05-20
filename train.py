import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import os
from datetime import datetime

from mnist_rot.data_loader_mnist_rot import build_mnist_rot_loader
from models.sfcnn import SFCNN

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
    parser.add_argument('--num_orientations', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--lr', type=float, default=0.015)
    parser.add_argument('--weight_decay', type=float, default=1e-7)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs(f"outputs/{timestamp}", exist_ok=True)
    
    # Load data using your existing dataloader
    train_loader, _, _ = build_mnist_rot_loader('train', args.batch_size)
    val_loader, _, _ = build_mnist_rot_loader('valid', args.batch_size)
    
    # Initialize model
    model = SFCNN(num_orientations=args.num_orientations).to(args.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Training loop
    best_val_acc = 0
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        # Train and validate
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, args.device)
        val_loss, val_acc = validate(model, val_loader, criterion, args.device)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"outputs/{timestamp}/best_model.pth")
            print("Saved new best model!")
    
    print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.2f}%")

if __name__ == "__main__":
    main()