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
        ax.imshow(imgi, cmap="gray")
        ax.set_title(channels[i])
    return fig



def plot_prediction__(ax, pred, color="blue", base=np.array([0, 0])):
    pred = np.cumsum(pred, axis=0)
    pred += base
    #pred = np.insert(pred, 0, base, axis=0)
    x, y = (pred[:, 0], pred[:, 1])
    ax.plot([base[0], x[0]], [base[1], y[0]], color=color, linestyle="dashed")
    ax.plot([x[0]], [y[0]], marker="o", markersize=10, markeredgecolor="black", markerfacecolor=color)
    ax.plot(x, y, color=color, linestyle="dashed")
    base = pred[0]
    return base

def plot_prediction_(ax, pred, color="blue", base=np.array([0, 0]), repeat_first=False):
    if len(pred.shape) == 2:
        pred = np.expand_dims(pred, axis=0)
    for i in range(pred.shape[0]):
        p = pred[i]
        if repeat_first:
            p = np.repeat(p[:1, :], len(p), axis=0)
        base = plot_prediction__(ax, p, color=color, base=base)
    return pred.shape[0]

def plot_image_(ax, x, permute=True):
    if permute:
        x = x.permute(-2, -1, -3)
    ax.imshow(x, extent=[
        -x.shape[1]/2., 
        x.shape[1]/2., 
        -x.shape[0]/2.,
        x.shape[0]/2.
    ], cmap="gray")

def plot_prediction(pred, y=None, pred_color="blue", y_color="orange", img=None, repeat_first_y=False):
    fig, ax = plt.subplots()

    if img is not None:
        plot_image_(ax, img)

    ax.plot([0], [0], marker="o", markersize=10, markeredgecolor="black", markerfacecolor="white")
    
    if y is not None:
        n_y = plot_prediction_(ax, y, color=y_color, repeat_first=repeat_first_y)
        #axes = axes + (n_y * ["ground_truth"])
    
    n_pred = plot_prediction_(ax, pred, color=pred_color)
    #axes = n_pred * ["prediction"] 
    
    #ax.legend(axes)
    
    return fig
    