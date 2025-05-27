import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from models.memnet import MembraneNet
from ISBI.dataloader import ISBIDataset
import os


def visualize_predictions(model, loader, device, num_images=5, save_dir="predictions"):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for idx, (data, _) in enumerate(loader):
            if idx >= num_images:
                break
            data = data.to(device)
            output = model(data)[..., 32:-32, 32:-32].squeeze().cpu().numpy()
            img = data.squeeze().cpu().numpy()

            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].imshow(img, cmap='gray')
            ax[0].set_title('Input Image')
            ax[1].imshow(output, cmap='gray')
            ax[1].set_title('Predicted Membrane Map')
            for a in ax:
                a.axis('off')
            plt.tight_layout()
            plt.savefig(f"{save_dir}/prediction_{idx}.png")
            plt.close()


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    test_dataset = ISBIDataset("ISBI/test-volume.tif", label_path=None, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = MembraneNet().to(device)
    model.load_state_dict(torch.load("membrane_net_epoch100.pth", map_location=device))
    visualize_predictions(model, test_loader, device)


if __name__ == '__main__':
    main()
