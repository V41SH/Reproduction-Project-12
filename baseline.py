import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF
from e2cnn import gspaces
from e2cnn import nn as enn
import random
import time

class RotatedMNIST(torch.utils.data.Dataset):
    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        img, label = self.base[idx]
        angle = random.uniform(0, 360)
        rotated_img = TF.rotate(img, angle, fill=0)
        return rotated_img, label

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
base_test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
rotated_test_dataset = RotatedMNIST(base_test_dataset)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(rotated_test_dataset, batch_size=1000, shuffle=False)

class Z2CNN(nn.Module):
    def __init__(self):
        super(Z2CNN, self).__init__()
        r2_act = gspaces.TrivialOnR2()

        self.input_type = enn.FieldType(r2_act, [r2_act.trivial_repr])

        self.block1 = enn.SequentialModule(
    enn.R2Conv(self.input_type, enn.FieldType(r2_act, 16 * [r2_act.trivial_repr]),
               kernel_size=5, padding=2, bias=False),
    enn.ReLU(enn.FieldType(r2_act, 16 * [r2_act.trivial_repr]), inplace=True),
    enn.PointwiseMaxPoolAntialiased(enn.FieldType(r2_act, 16 * [r2_act.trivial_repr]),
                                    kernel_size=3, sigma=0.66, stride=2)
)

        self.block2 = enn.SequentialModule(
            enn.R2Conv(enn.FieldType(r2_act, 16 * [r2_act.trivial_repr]),
                       enn.FieldType(r2_act, 32 * [r2_act.trivial_repr]),
                       kernel_size=5, padding=2, bias=False),
            enn.ReLU(enn.FieldType(r2_act, 32 * [r2_act.trivial_repr]), inplace=True),
            enn.PointwiseMaxPoolAntialiased(enn.FieldType(r2_act, 32 * [r2_act.trivial_repr]), kernel_size=3, sigma=0.66, stride=2)
        )

        self.gpool = enn.GroupPooling(enn.FieldType(r2_act, 32 * [r2_act.trivial_repr]))
        self.fc1 = nn.Linear(32 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = enn.GeometricTensor(x, self.input_type)
        x = self.block1(x)
        x = self.block2(x)
        x = self.gpool(x)
        x = x.tensor
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Z2CNN().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    total_loss = 0
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch {epoch}: Avg loss = {total_loss / len(train_loader):.4f}")

def test(model, device, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    error_rate = 100. * (1 - correct / total)
    print(f"Test Error on Rotated MNIST: {error_rate:.2f}%")
    return error_rate

start_time = time.time()
for epoch in range(1, 20):  # Increase to 20 for better accuracy
    train(model, device, train_loader, optimizer, epoch)
test_error = test(model, device, test_loader)
print(f"Finished in {time.time() - start_time:.1f}s")
