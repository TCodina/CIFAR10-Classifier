"""
This file contains the main class for the CIFAR-10 dataset and for the pre-processing transformations.
TODO: The dataset class assumes all data is already stored in a local variable. In more realistic scenarios where
the dataset cannot be stored all at once, this class should read each example only when calling the __getitem__ method
"""

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset


class CIFARDataset(Dataset):
    """
    Custom CIFAR10 dataset
    """

    def __init__(self, data_dict, transform=None):
        self.data_dict = data_dict
        self.transform = transform

    def __len__(self):
        return len(self.data_dict['labels'])

    def __getitem__(self, idx):
        img = self.data_dict['imgs'][idx]
        label = self.data_dict['labels'][idx]

        if self.transform:
            img, label = self.transform(img, label)

        return img, label


class CustomTransform:
    def __init__(self, norm_mean, norm_std, resize=None, use_augmentation=False):

        # Mean and std for standarization (from CIFAR10 or ImageNet training set)
        self.norm_mean = norm_mean
        self.norm_std = norm_std

        trans_list = []
        
        # Resize and data augmentation transformations
        if resize is not None:
            trans_list += [transforms.ToPILImage(),
                           transforms.Resize(256)]
            if use_augmentation:
                trans_list += [transforms.RandomCrop(resize), 
                               transforms.RandomHorizontalFlip(p=0.5)]
            else:
                trans_list += [transforms.CenterCrop(resize)]
        
        # Common transformations to instances
        trans_list += [transforms.ToTensor(),
                       transforms.Normalize(mean=self.norm_mean, std=self.norm_std)]
        
        self.image_transformation = transforms.Compose(trans_list)

    def __call__(self, img, label):
        img_t = self.image_transformation(img)
        label_t = torch.tensor(label)
        return img_t, label_t

