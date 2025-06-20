import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.sfcnn import SFCNN
import matplotlib.pyplot as plt


def test(model, device, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    error_rate = 1 - correct / total
    error_rate *= 100
    print(f"Test error rate: {error_rate:.4f}")
    return error_rate


if __name__ == '__main__':
    batch_size = 128
    model_samples = {
        'mnist750new': 750,
        'mnist3knew': 3000,
        'mnist12knew': 12000
    }
    model_orientations = [4, 8, 16, 20]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    results = {samples: [] for samples in model_samples.keys()}

    for sample_size in model_samples:
        for orientation in model_orientations:
            model = SFCNN(num_orientations=orientation)
            model_name = f"{orientation}best_model.pth"
            model.load_state_dict(
                torch.load(f"/kaggle/input/{sample_size}/pytorch/default/1/{model_name}", map_location=device))
            model.to(device)

            error_percent = test(model, device, test_loader)
            results[sample_size].append(error_percent)

    # Plotting
    plt.figure(figsize=(8, 6))
    for sample_size, error_list in results.items():
        plt.plot(model_orientations, error_list, marker='o', label=str(sample_size))

    plt.xlabel("Number of angles Λ")
    plt.ylabel("Test error [%]")
    plt.yscale("log")
    plt.legend(title="num samples")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("test_error_vs_orientations.png")
    plt.show()

