import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

#switch to seaborn defaults
sns.set_theme(style='darkgrid')

def plot_images(images, targets, n_plot=30):
    n_rows = n_plot // 6 + ((n_plot % 6) > 0)
    fig, axes = plt.subplots(n_rows, 6, figsize=(9, 1.5 * n_rows))
    axes = np.atleast_2d(axes)

    for i, (image, target) in enumerate(zip(images[:n_plot], targets[:n_plot])):
        row, col = i // 6, i % 6
        ax = axes[row, col]
        ax.set_title('#{} - Label:{}'.format(i, target), {'size': 12})
        # plot filter channel in grayscale
        ax.imshow(image.squeeze(), cmap='gray', vmin=0, vmax=1)

    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.label_outer()

    plt.tight_layout()
    return fig

def show_weights(weights):
    fig, axs = plt.subplots(1, 5, figsize=(10, 4))

    for i, m in enumerate(weights):
        axs[i].imshow(m.reshape(-1, 5).tolist(), cmap='gray')
        axs[i].grid(False)
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        axs[i].set_title(r'$w_{0' + str(i) + '}$')

    fig.suptitle('Hidden Layer #0')
    fig.subplots_adjust(top=0.6)
    fig.tight_layout()

def gen_img(start, target, fill=1, img_size=10):
    # Generates empty image
    img = np.zeros((img_size, img_size), dtype=float)

    start_row, start_col = None, None

    if start > 0:
        start_row = start
    else:
        start_col = np.abs(start)

    if target == 0:
        if start_row is None:
            img[:, start_col] = fill
        else:
            img[start_row, :] = fill
    else:
        if start_col == 0:
            start_col = 1

        if target == 1:
            if start_row is not None:
                up = (range(start_row, -1, -1),
                      range(0, start_row + 1))
            else:
                up = (range(img_size - 1, start_col - 1, -1),
                      range(start_col, img_size))
            img[up] = fill
        else:
            if start_row is not None:
                down = (range(start_row, img_size, 1),
                        range(0, img_size - start_row))
            else:
                down = (range(0, img_size - 1 - start_col + 1),
                        range(start_col, img_size))
            img[down] = fill

    return 255 * img.reshape(1, img_size, img_size)

def skewed_seq(r):
    out = []
    for i in range(1,r):
        c = random.choices(range(0, 3), weights=(3,1,1))[0]
        out.append(c)
    return out

def generate_dataset(img_size=10, n_images=100, binary=True, seed=17):
    np.random.seed(seed)

    starts = np.random.randint(-(img_size - 1), img_size, size=(n_images,))
    targets = np.array(skewed_seq(n_images))

    images = np.array([gen_img(s, t, img_size=img_size)
                       for s, t in zip(starts, targets)], dtype=np.uint8)

    if binary:
        targets = (targets > 0).astype(int)

    return images, targets
