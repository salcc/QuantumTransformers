import matplotlib.pyplot as plt
import numpy as np


def plot_image(image, abs_log=False):
    """Plots an image with one subplot per channel"""
    num_channels = image.shape[2]
    fig, axs = plt.subplots(1, num_channels, figsize=(num_channels * 3, 3))
    if num_channels == 1:
        axs = [axs]
    for i, ax in enumerate(axs):
        if abs_log:
            ax.imshow(np.log(np.abs(image[:, :, i]) + 1e-6))
        else:
            ax.imshow(image[:, :, i])
    plt.show()
