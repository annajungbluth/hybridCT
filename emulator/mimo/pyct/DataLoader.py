import torch
import numpy as np
from torch.utils.data import Dataset

class Cloud_Dataset(Dataset):
    def __init__(self, X, y, transform_image=None):
        self.X = X
        self.y = y
        self.data = dict(
            image=X, 
            label=y,
        )
        self.transform_image = transform_image

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        image = self.X[idx,...]
        label = self.y[idx]

        if self.transform_image:
            image = np.moveaxis(image,0,2).astype(np.float32)
            label = np.moveaxis(label,0,2).astype(np.float32)
            #Dummy channel to ensure extraction of label when horizontal flip occurs
            dummy = np.zeros((label.shape[0], label.shape[1],1), dtype=np.int32) - 999
            channels = image.shape[2]
            cat_data = np.concatenate((image,label, dummy), axis=2).astype(np.float32)
            transformed = self.transform_image(cat_data)
            if transformed[0].min() <= -999:
                image = transformed[1+channels:,...]
                label = transformed[1:1+channels,...]
            else:
                image = transformed[0:channels,...]
                label = transformed[channels:-1,...]
            if label.ndim < 3:
                label = torch.unsqueeze(label, dim=0)

        

        sample = {'image':image, 'label':label} #(image, label)
        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        image, label = sample
        img_tensor = torch.as_tensor(image, dtype=torch.float)
        label_tensor = torch.as_tensor(label, dtype=torch.float)
        return img_tensor, label_tensor

 
 
class MultiAngleRandomHorizontalFlip(torch.nn.Module):
    """ Specialization of torchvision RandomHorizontalFlip.

    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, img):
        """
        Args:
            img (Tensor): Image to be flipped.

        Returns:
            Tensor: Randomly flipped image. If mutli channel (multi-angle), 
                channel dimension is also flipped.
        """
        if torch.rand(1) < self.p:
            #Horizontal and channel
            return img.flip(-1,-3)
        return img


