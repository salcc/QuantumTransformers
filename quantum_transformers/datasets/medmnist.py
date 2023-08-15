import os

import torch
import torch.utils.data
import torchvision
import medmnist

from quantum_transformers.datasets import datasets_to_dataloaders

def get_medmnist_dataloaders(dataset: str, root: str = '~/data', download: bool = True, **dataloader_kwargs) \
        -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Returns dataloaders for a MedMNIST dataset"""
    root = os.path.expanduser(root)
    assert dataset in medmnist.INFO, f'Unknown MedMNIST dataset: {dataset}.\nValid datasets: {list(medmnist.INFO.keys())}'
    info = medmnist.INFO[dataset]
    DataClass = getattr(medmnist, info['python_class'])
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    train_dataset = DataClass(split='train', transform=transform, download=download)
    valid_dataset = DataClass(split='test', transform=transform, download=download)
    return datasets_to_dataloaders(train_dataset, valid_dataset, **dataloader_kwargs)
