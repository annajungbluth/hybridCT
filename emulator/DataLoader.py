import torch
import numpy as np
from torch.utils.data import Dataset

class Cloud_Dataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        image = self.X[idx,...]
        label = self.y[idx]
        sample = (image, label)
        if self.transform:
            sample = self.transform(sample)
        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image, label = sample
        img_tensor = torch.as_tensor(image, dtype=torch.float)
        label_tensor = torch.as_tensor(label, dtype=torch.float)
        return img_tensor, label_tensor


