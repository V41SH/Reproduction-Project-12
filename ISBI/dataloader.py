from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import tifffile
import random

class ISBIDataset(Dataset):
    def __init__(self, volume_path, label_path=None, patch_size=256, pad=32, transform=None):
        self.volume = tifffile.imread(volume_path)
        self.label = tifffile.imread(label_path) if label_path is not None else None
        self.patch_size = patch_size
        self.pad = pad
        self.transform = transform

        self.indices = list(range(self.volume.shape[0]))  # Each slice is a separate image

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        slice_img = self.volume[idx]
        slice_img = np.pad(slice_img, self.pad, mode='reflect') # reflect padding

        image = Image.fromarray(slice_img.astype(np.uint8))

        if self.label is not None:
            slice_label = self.label[idx]
            slice_label = np.pad(slice_label, self.pad, mode='reflect')
            label = Image.fromarray((slice_label > 0).astype(np.float32))  # binarize
        else:
            label = Image.new('F', (image.width, image.height))

        if self.transform:
            seed = random.randint(0, 99999)
            random.seed(seed)
            image = self.transform(image)
            random.seed(seed)
            label = self.transform(label)

        return image, label
