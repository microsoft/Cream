# --------------------------------------------------------
# reference: https://github.com/crj1998/pruning/tree/master
# --------------------------------------------------------
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap

import numpy as np


color_list = ['pink', 'deepskyblue']

my_cmap = LinearSegmentedColormap.from_list('custom', color_list)

cm.register_cmap(cmap=my_cmap)


def plot(heads, intermediates, name):
    fig, ax = plt.subplots(1, 2, facecolor='white', figsize=(
        10, 4), dpi=120, gridspec_kw={'width_ratios': [1.15, 3]})

    heads_num = heads.shape[1]
    ax[0].matshow(heads, cmap="custom", vmin=0.0, vmax=1.0)
    ax[0].set_xlabel("Heads")
    ax[0].set_ylabel("Layer")
    ax[0].set_xticks([i for i in range(heads_num)], [str(i + 1)
                     for i in range(heads_num)])
    ax[0].set_yticks([i for i in range(12)], [str(i + 1) for i in range(12)])
    # Minor ticks
    ax[0].set_xticks([i - 0.5 for i in range(heads_num)], minor=True)
    ax[0].set_yticks([i - 0.5 for i in range(12)], minor=True)
    ax[0].xaxis.tick_bottom()
    ax[0].tick_params('both', length=0, width=0, which='both')

    # Gridlines based on minor ticks
    ax[0].grid(which='minor', color='w', linestyle='-', linewidth=1)
    ax[0].set_title('MHAs')

    channel = intermediates.shape[1] / 4
    intermediates = intermediates.repeat(100, axis=0)
    ax[1].matshow(intermediates, cmap="custom", vmin=0.0, vmax=1.0)
    ax[1].set_xlabel("FFNs channels")

    ax[1].set_xticks([i * channel for i in range(1, 5)],
                     [f'{i}.0x' for i in range(1, 5)])
    ax[1].set_yticks([i * 100 + 50 for i in range(12)],
                     [str(i + 1) for i in range(12)])
    ax[1].set_yticks([i * 100 for i in range(12)], minor=True)

    # Minor ticks

    ax[1].xaxis.tick_bottom()
    ax[1].yaxis.tick_right()

    ax[1].tick_params('both', length=0, width=0, which='both')

    # Gridlines based on minor ticks
    ax[1].grid(which='minor', axis='y', color='w', linestyle='-', linewidth=1)
    ax[1].set_title('FFNs')

    fig.tight_layout()

    fig.suptitle(name)

    return fig
