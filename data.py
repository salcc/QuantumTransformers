import h5py
import torch.utils.data
import torchvision.datasets

def datasets_to_dataloaders(train_dataset, valid_dataset, **dataloader_kwargs):
    """Returns dataloaders for the given datasets"""
    train_dataloader = torch.utils.data.DataLoader(train_dataset, **dataloader_kwargs)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, **dataloader_kwargs)
    return train_dataloader, valid_dataloader


def get_mnist_dataloaders(download=True, **dataloader_kwargs):
    """Returns dataloaders for the MNIST digits dataset (computer vision, 10-class classification)"""
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = torchvision.datasets.MNIST('data', train=True, download=download, transform=transform)
    valid_dataset = torchvision.datasets.MNIST('data', train=False, download=download, transform=transform)
    return datasets_to_dataloaders(train_dataset, valid_dataset, **dataloader_kwargs)


def load_hdf5_datasets(root, train, test, n=None, X_train_key='X', y_train_key='y', X_test_key='X', y_test_key='y'):
    """Loads the given HDF5 datasets from the given root directory and returns them as numpy arrays.
    
    The datasets are assumed to be in the following format:
    - X: a dataset of features, shape (n_samples, n_features...)
    - y: a dataset of labels, shape (n_samples,)

    If n is not None, only the first n samples are loaded from each dataset (useful for testing in low-memory environments).
    """
    with h5py.File(f"{root}/{train}", 'r') as f:
        if n is not None:
            X_train, y_train = f[X_train_key][:n], f[y_train_key][:n]
        else:
            X_train, y_train = f[X_train_key][:], f[y_train_key][:]
    with h5py.File(f"{root}/{test}", 'r') as f:
        if n is not None:
            X_test, y_test = f[X_test_key][:n], f[y_test_key][:n]
        else:
            X_test, y_test = f[X_test_key][:], f[y_test_key][:]

    return X_train, y_train, X_test, y_test


def get_electron_photon_dataloaders(n=None, **dataloader_kwargs):
    """Returns dataloaders for the electron-photon dataset (computer vision - particle physics, binary classification)"""
    X_train, y_train, X_test, y_test = load_hdf5_datasets('data', 'electron-photon_train-set_n488000.hdf5', 'electron-photon_test-set_n10000.hdf5', n)
    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    valid_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    return datasets_to_dataloaders(train_dataset, valid_dataset, **dataloader_kwargs)

def get_quark_gluon_dataloaders(n=None, **dataloader_kwargs):
    """Returns dataloaders for the quark-gluon dataset (computer vision - particle physics, binary classification)"""
    X_train, y_train, X_test, y_test = load_hdf5_datasets('data', 'quark-gluon_train-set_n139306.hdf5', 'quark-gluon_test-set_n10000.hdf5', n, X_train_key='X_jets')
    train_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    valid_dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    return datasets_to_dataloaders(train_dataset, valid_dataset, **dataloader_kwargs)

def get_imdb_dataloaders(download=True, **dataloader_kwargs):
    """Returns dataloaders for the IMDB sentiment analysis dataset (natural language processing, binary classification)"""
    train_dataset = torchvision.datasets.IMDB('data', train=True, download=download)
    valid_dataset = torchvision.datasets.IMDB('data', train=False, download=download)
    raise NotImplementedError # TODO: add transforms (tokenize, etc.)
    return datasets_to_dataloaders(train_dataset, valid_dataset, **dataloader_kwargs)