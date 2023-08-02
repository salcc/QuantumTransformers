
import torch
import torch.utils.data


def datasets_to_dataloaders(train_dataset: torch.utils.data.Dataset, valid_dataset: torch.utils.data.Dataset, **dataloader_kwargs) \
        -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Returns dataloaders for the given datasets"""
    train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, **dataloader_kwargs)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, **dataloader_kwargs)
    return train_dataloader, valid_dataloader
