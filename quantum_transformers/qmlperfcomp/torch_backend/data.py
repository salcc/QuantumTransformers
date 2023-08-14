import os

import torch
import torch.utils.data
import torchvision

from quantum_transformers.qmlperfcomp.swiss_roll import make_swiss_roll_dataset


def datasets_to_dataloaders(train_dataset: torch.utils.data.Dataset, valid_dataset: torch.utils.data.Dataset, **dataloader_kwargs) \
        -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Returns dataloaders for the given datasets"""
    train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, **dataloader_kwargs)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, **dataloader_kwargs)
    return train_dataloader, valid_dataloader


def get_swiss_roll_dataloaders(dataset_size: int = 500, train_frac: float = 0.8, **dataloader_kwargs):
    """Returns dataloaders for the Swiss roll dataset (3 features, binary classification)"""
    train_inputs, train_labels, eval_inputs, eval_labels = make_swiss_roll_dataset(n_points=dataset_size, train_frac=train_frac)
    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(train_inputs), torch.from_numpy(train_labels))
    valid_dataset = torch.utils.data.TensorDataset(torch.from_numpy(eval_inputs), torch.from_numpy(eval_labels))
    return datasets_to_dataloaders(train_dataset, valid_dataset, **dataloader_kwargs)


def get_mnist_dataloaders(root: str = '~/data', **dataloader_kwargs) \
        -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Returns dataloaders for the MNIST digits dataset (computer vision, 10-class classification)"""
    root = os.path.expanduser(root)
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = torchvision.datasets.MNIST(root, train=True, download=True, transform=transform)
    valid_dataset = torchvision.datasets.MNIST(root, train=False, download=True, transform=transform)

    return datasets_to_dataloaders(train_dataset, valid_dataset, **dataloader_kwargs)
