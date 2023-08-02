import os
import tarfile

import gdown
import numpy as np
import torch
import torch.utils.data
import torchvision

from quantum_transformers.datasets import datasets_to_dataloaders


def download_dataset(root: str, name: str, gdrive_id: str, remove_archive: bool = True, verbose: bool = True) -> None:
    """Downloads a dataset from Google Drive and extracts it if it does not already exist"""
    if os.path.exists(f'{root}/{name}'):
        if verbose:
            print(f'{root}/{name} already exists, skipping download')
        return
    os.makedirs(f'{root}/{name}', exist_ok=True)
    gdown.download(id=gdrive_id, output=f'{root}/{name}.tar.xz', quiet=not verbose)
    with tarfile.open(f'{root}/{name}.tar.xz') as f:
        print(f'Extracting {name}.tar.xz to {root}...')
        f.extractall(f'{root}')
    if remove_archive:
        os.remove(f'{root}/{name}.tar.xz')


def npy_loader(path: str) -> torch.Tensor:
    """Loads a .npy file as a PyTorch tensor"""
    return torch.from_numpy(np.load(path))


def get_electron_photon_dataloaders(root: str = '~/data', download: bool = True, **dataloader_kwargs) \
        -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Returns dataloaders for the electron-photon dataset (computer vision - particle physics, binary classification)"""
    root = os.path.expanduser(root)
    if download:
        download_dataset(root, 'electron-photon', '1VAqGQaMS5jSWV8gTXw39Opz-fNMsDZ8e')
    train_dataset = torchvision.datasets.DatasetFolder(root=f'{root}/electron-photon/train', loader=npy_loader, extensions=('.npy',))
    valid_dataset = torchvision.datasets.DatasetFolder(root=f'{root}/electron-photon/test', loader=npy_loader, extensions=('.npy',))
    return datasets_to_dataloaders(train_dataset, valid_dataset, **dataloader_kwargs)


def get_quark_gluon_dataloaders(root: str = '~/data', download: bool = True, **dataloader_kwargs) \
        -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Returns dataloaders for the quark-gluon dataset (computer vision - particle physics, binary classification)"""
    root = os.path.expanduser(root)
    if download:
        download_dataset(root, 'quark-gluon', '1G6HJKf3VtRSf7JLms2t1ofkYAldOKMls')
    train_dataset = torchvision.datasets.DatasetFolder(root=f'{root}/quark-gluon/train', loader=npy_loader, extensions=('.npy',))
    valid_dataset = torchvision.datasets.DatasetFolder(root=f'{root}/quark-gluon/test', loader=npy_loader, extensions=('.npy',))
    return datasets_to_dataloaders(train_dataset, valid_dataset, **dataloader_kwargs)
