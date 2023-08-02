import gc

import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_image(image, abs_log=False):
    """Plots an image with one subplot per channel"""
    num_channels = image.shape[2]
    fig, axs = plt.subplots(1, num_channels, figsize=(num_channels*3, 3))
    if num_channels == 1:
        axs = [axs]
    for i, ax in enumerate(axs):
        if abs_log:
            ax.imshow(np.log(np.abs(image[:, :, i]) + 1e-6))
        else:
            ax.imshow(image[:, :, i])
    plt.show()


def delete_variables(*variables):
    """Deletes variables from memory, empties the GPU cache, and runs garbage collection"""
    for var in variables:
        del var
    torch.cuda.empty_cache()
    gc.collect()
