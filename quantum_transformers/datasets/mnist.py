import os

import torch
import torch.utils.data
import torchvision

from quantum_transformers.datasets import datasets_to_dataloaders


def get_mnist_dataloaders(root: str = '~/data', download: bool = True, **dataloader_kwargs) \
        -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Returns dataloaders for the MNIST digits dataset (computer vision, 10-class classification)"""
    root = os.path.expanduser(root)
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = torchvision.datasets.MNIST(root, train=True, download=download, transform=transform)
    valid_dataset = torchvision.datasets.MNIST(root, train=False, download=download, transform=transform)
    return datasets_to_dataloaders(train_dataset, valid_dataset, **dataloader_kwargs)
