import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader
from dataloader import ISBIDataset

def main():
    volume_path = 'ISBI/train-volume.tif'
    label_path = 'ISBI/train-labels.tif'

    # Define a basic transform if needed
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Initialize dataset
    dataset = ISBIDataset(volume_path, label_path, transform=transform)

    print(f"Number of images in dataset: {len(dataset)}")

    # Load using DataLoader for convenience (optional)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Visualize first few samples
    for i, (image, label) in enumerate(dataloader):
        print(f"\nSample {i}:")
        print(f"  Image shape: {image.shape}, dtype: {image.dtype}")
        print(f"  Label shape: {label.shape}, dtype: {label.dtype}")

        # Show image and label
        img_np = image.squeeze().numpy()
        lbl_np = label.squeeze().numpy()

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(img_np, cmap='gray')
        axs[0].set_title('Image')
        axs[1].imshow(lbl_np, cmap='gray')
        axs[1].set_title('Label')
        plt.show()

        if i >= 4:  # Show only first 5
            break

if __name__ == "__main__":
    main()
