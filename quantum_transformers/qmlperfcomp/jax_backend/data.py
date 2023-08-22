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
