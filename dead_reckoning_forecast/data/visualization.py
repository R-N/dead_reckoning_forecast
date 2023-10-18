from .. import constants
import numpy as np
import matplotlib.pyplot as plt

def plot_frame(img, channels=constants.channels, permute=True):
    if permute:
        img = img.permute(-2, -1, -3)
    fig, axs = plt.subplots(2, 2)
    fig.tight_layout()
    for i in range(img.shape[-1]):
        imgi = img[:, :, i]
        ax = axs[i//2, i%2]
        ax.imshow(imgi)
        ax.set_title(channels[i])
    return fig


def plot_prediction(pred, y=None):
    fig, ax = plt.subplots()
    
    pred = np.cumsum(pred, axis=0)
    ax.plot(pred[:, 0], pred[:, 1])
    axes = ["prediction"]
    
    if y is not None:
        y = np.cumsum(pred, axis=0)
        ax.plot(y[:, 0], y[:, 1])
        axes.append("ground_truth")
    
    ax.legend(axes)
    
    return fig
    