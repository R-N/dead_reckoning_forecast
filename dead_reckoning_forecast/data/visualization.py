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



def plot_prediction__(ax, pred, color="blue", base=np.array([0, 0])):
    pred = np.cumsum(pred, axis=0)
    pred += base
    ax.plot(pred[:, 0], pred[:, 1], color=color, linestyle="dashed")
    base = pred[0]
    return base

def plot_prediction_(ax, pred, color="blue", base=np.array([0, 0])):
    if len(pred.shape) == 2:
        pred = np.expand_dims(pred, axis=0)
    for i in range(pred.shape[0]):
        base = plot_prediction__(ax, pred[i], color=color, base=base)
    return base

def plot_prediction(pred, y=None, pred_color="blue", y_color="orange"):
    fig, ax = plt.subplots()
    
    plot_prediction_(ax, pred, color=pred_color)
    axes = ["prediction"]
    
    if y is not None:
        plot_prediction_(ax, y, color=y_color)
        axes.append("ground_truth")
    
    ax.legend(axes)
    
    return fig
    