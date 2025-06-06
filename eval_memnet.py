import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from models.memnet import MembraneNet
from ISBI.dataloader import ISBIDataset
import os

def visualize_predictions(model, loader, device, num_images=30, save_dir="predictions"):
    import numpy as np
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        for idx, (data, _) in enumerate(loader):
            if idx >= num_images:
                break
            data = data.to(device)
            output = model(data)[..., 16:-16, 16:-16].squeeze().cpu().numpy()
            img = data.squeeze().cpu().numpy()

            # Normalize output to [0, 1]
            output_norm = (output - output.min()) / (output.max() - output.min())

            # Invert: darker areas become higher values
            inverted = 1.0 - output_norm

            # Threshold: keep only darkest areas
            binary_mask = (inverted > 0.58).astype(float)  # Adjust threshold as needed
            binary_mask = 1 - binary_mask
            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
            ax[0].imshow(img, cmap='gray')
            ax[0].set_title('Input Image')
            ax[1].imshow(output_norm, cmap='gray')
            ax[1].set_title('Predicted Membrane Map')
            ax[2].imshow(binary_mask, cmap='gray')
            ax[2].set_title('Binary Mask (Darkest Areas)')
            for a in ax:
                a.axis('off')
            plt.tight_layout()
            plt.savefig(f"{save_dir}/prediction_binary_{idx}.png")
            plt.close()

# def visualize_predictions(model, loader, device, num_images=5, save_dir="predictions"):
#     os.makedirs(save_dir, exist_ok=True)
#     model.eval()
#     with torch.no_grad():
#         for idx, (data, _) in enumerate(loader):
#             if idx >= num_images:
#                 break
#             data = data.to(device)
#             output = model(data)[..., 32:-32, 32:-32].squeeze().cpu().numpy()


#             img = data.squeeze().cpu().numpy()

#             fig, ax = plt.subplots(1, 2, figsize=(10, 5))
#             ax[0].imshow(img, cmap='gray')
#             ax[0].set_title('Input Image')
#             ax[1].imshow(output, cmap='gray')
#             ax[1].set_title('Predicted Membrane Map')
#             for a in ax:
#                 a.axis('off')
#             plt.tight_layout()
#             plt.savefig(f"{save_dir}/prediction_{idx}.png")
#             plt.close()

from PIL import Image
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os

def visualize_from_image_path(model, image_path, device, save_path="predictions/prediction_single.png"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Load and preprocess image
    image = Image.open(image_path).convert("L")  # Convert to grayscale
    transform = transforms.ToTensor()
    image_tensor = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    # Run inference
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)[..., 16:-16, 16:-16].squeeze().cpu().numpy()
        img = image_tensor.squeeze().cpu().numpy()

        # Normalize and process
        output_norm = (output - output.min()) / (output.max() - output.min())
        inverted = 1.0 - output_norm
        binary_mask = (inverted > 0.6).astype(float)
        binary_mask = 1 - binary_mask

        # Plot
        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        ax[0].imshow(img, cmap='gray')
        ax[0].set_title('Input Image')
        ax[1].imshow(output_norm, cmap='gray')
        ax[1].set_title('Predicted Membrane Map')
        ax[2].imshow(binary_mask, cmap='gray')
        ax[2].set_title('Binary Mask (Darkest Areas)')
        for a in ax:
            a.axis('off')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MembraneNet().to(device)
    model.load_state_dict(torch.load("membrane_net_epoch40.pth", map_location=device), strict=False)

    # Path to your test image
    image_path = "/home/salonisaxena/work/Q4/FUNML/Reproduction-Project-12/isbi-datasets-master/data/images/train-volume29.jpg"  # Can also point to a .png or .jpg if available

    visualize_from_image_path(model, image_path, device)

# def main():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     transform = transforms.Compose([
#         transforms.ToTensor()
#     ])

#     test_dataset = ISBIDataset("ISBI/test-volume.tif", label_path=None, transform=transform)
#     test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

#     model = MembraneNet().to(device)
#     model.load_state_dict(torch.load("membrane_net_epoch100.pth", map_location=device), strict=False)



#     visualize_predictions(model, test_loader, device)


if __name__ == '__main__':
    main()