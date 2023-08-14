from sklearn.datasets import make_swiss_roll
import matplotlib.pyplot as plt
import numpy as np

def make_swiss_roll_dataset(n_points: int = 500, train_frac: float = 0.8, plot=False):
    inputs, t = make_swiss_roll(n_samples=n_points, noise=0.1, random_state=0)
    inputs = inputs.astype(np.float32)
    labels = np.expand_dims(np.where(t < np.mean(t), 0, 1), axis=1)

    if plot:
        ax = plt.axes(projection='3d')
        ax.scatter(inputs[:, 0], inputs[:, 1], inputs[:, 2], c=labels, cmap='RdYlGn');
        ax.view_init(azim=-75, elev=3)
        ax.set_title('Swiss roll dataset')
        plt.show()

    # Shuffle and split the dataset
    shuffled_indices = np.random.permutation(inputs.shape[0])
    train_size = int(train_frac * inputs.shape[0])

    train_inputs = inputs[shuffled_indices[:train_size]]
    train_labels = labels[shuffled_indices[:train_size]]

    valid_inputs = inputs[shuffled_indices[train_size:]]
    valid_labels = labels[shuffled_indices[train_size:]]

    return train_inputs, train_labels, valid_inputs, valid_labels
