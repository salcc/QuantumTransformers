import tensorflow_datasets as tfds
import tensorflow as tf
# Ensure TF does not see GPU and grab all GPU memory.
tf.config.set_visible_devices([], device_type='GPU')

from quantum_transformers.qmlperfcomp.swiss_roll import make_swiss_roll_dataset


def datasets_to_dataloaders(train_dataset, valid_dataset, batch_size):
    # Shuffle train dataset
    train_dataset = train_dataset.shuffle(train_dataset.cardinality(), reshuffle_each_iteration=True)

    # Batch and prefetch
    train_dataloader = train_dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
    valid_dataloader = valid_dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)

    # Convert to numpy
    return tfds.as_numpy(train_dataloader), tfds.as_numpy(valid_dataloader)


def get_swiss_roll_dataloaders(plot: bool = False, dataset_size: int = 500, train_frac: float = 0.8, batch_size: int = 10):
    train_inputs, train_labels, valid_inputs, valid_labels = make_swiss_roll_dataset(n_points=dataset_size, train_frac=train_frac, plot=plot)

    # Convert the splits to TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((train_inputs, train_labels))
    valid_dataset = tf.data.Dataset.from_tensor_slices((valid_inputs, valid_labels))

    return datasets_to_dataloaders(train_dataset, valid_dataset, batch_size)


def get_mnist_dataloaders(root: str = '~/data', plot: bool = False, batch_size: int = 10):
    def normalize_image(image, label):
        image = tf.cast(image, tf.float32) / 255.0
        return (image - 0.1307) / 0.3081, label

    # Load datasets
    train_dataset = tfds.load(name='mnist', split='train', as_supervised=True, data_dir=root, shuffle_files=True)
    valid_dataset = tfds.load(name='mnist', split='test', as_supervised=True, data_dir=root)

    if plot:
        import matplotlib.pyplot as plt
        from mpl_toolkits.axes_grid1 import ImageGrid
        fig = plt.figure(figsize=(5, 4))
        grid = ImageGrid(fig, 111, nrows_ncols=(4, 5), axes_pad=0.1)
        for i, (image, label) in enumerate(train_dataset.take(20)):
            grid[i].imshow(image.numpy()[:, :, 0], cmap='gray')
        fig.suptitle('MNIST dataset')
        fig.show()

    # Normalize
    train_dataset = train_dataset.map(normalize_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    valid_dataset = valid_dataset.map(normalize_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)

    return datasets_to_dataloaders(train_dataset, valid_dataset, batch_size)


# Ideally we would use the PyTorch dataloaders (see below), but they do not work with num_workers > 0 (so they are slow): https://github.com/google/jax/issues/9190.

# import os

# import torch
# import torch.utils.data
# import torchvision
# import numpy as np
# import jax.numpy as jnp

# # See https://jax.readthedocs.io/en/latest/notebooks/Neural_Network_and_Data_Loading.html.

# def numpy_collate(batch):
#     if isinstance(batch[0], np.ndarray):
#         return np.stack(batch)
#     elif isinstance(batch[0], (tuple, list)):
#         transposed = zip(*batch)
#         return [numpy_collate(samples) for samples in transposed]
#     else:
#         return np.array(batch)


# class NumpyLoader(torch.utils.data.DataLoader):
#     def __init__(self, dataset, batch_size=1,
#                  shuffle=False, sampler=None,
#                  batch_sampler=None, num_workers=0,
#                  pin_memory=False, drop_last=False,
#                  timeout=0, worker_init_fn=None):
#         super(self.__class__, self).__init__(dataset,
#                                              batch_size=batch_size,
#                                              shuffle=shuffle,
#                                              sampler=sampler,
#                                              batch_sampler=batch_sampler,
#                                              num_workers=num_workers,
#                                              collate_fn=numpy_collate,
#                                              pin_memory=pin_memory,
#                                              drop_last=drop_last,
#                                              timeout=timeout,
#                                              worker_init_fn=worker_init_fn)


# class NormalizeAndCast:
#     def __call__(self, pic):
#         pic = np.array(pic, dtype=np.float32)
#         pic = (pic - 0.1307) / 0.3081  # Normalize
#         if len(pic.shape) == 2:
#             pic = pic[:, :, np.newaxis]  # Add channel dimension
#         pic = jnp.array(pic, dtype=jnp.float32)  # Cast
#         return pic


# def datasets_to_dataloaders(train_dataset: torch.utils.data.Dataset, valid_dataset: torch.utils.data.Dataset, **dataloader_kwargs) \
#         -> tuple[NumpyLoader, NumpyLoader]:
#     """Returns dataloaders for the given datasets"""
#     train_dataloader = NumpyLoader(train_dataset, shuffle=True, **dataloader_kwargs)
#     valid_dataloader = NumpyLoader(valid_dataset, **dataloader_kwargs)
#     return train_dataloader, valid_dataloader


# def get_mnist_dataloaders(root: str = '~/data', download: bool = True, **dataloader_kwargs) \
#         -> tuple[NumpyLoader, NumpyLoader]:
#     """Returns dataloaders for the MNIST digits dataset (computer vision, 10-class classification)"""
#     root = os.path.expanduser(root)
#     train_dataset = torchvision.datasets.MNIST(root, train=True, download=download, transform=NormalizeAndCast())
#     valid_dataset = torchvision.datasets.MNIST(root, train=False, download=download, transform=NormalizeAndCast())
#     return datasets_to_dataloaders(train_dataset, valid_dataset, **dataloader_kwargs)
