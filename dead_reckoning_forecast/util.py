import shutil
import os
import torch
from PIL import Image
import numpy as np

from moviepy.editor import VideoFileClip

Tensor = torch.Tensor
DEFAULT_DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

def remake_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    while os.path.exists(path):
        pass
    while True:
        try:
            os.makedirs(path, exist_ok=True)
            if not os.path.exists(path):
                continue
            break
        except PermissionError:
            continue

def stack_samples(samples):
    samples = list(zip(*samples))
    samples = [torch.stack(x) for x in samples]
    return samples

def to_tensor(x, Tensor=Tensor):
    if not Tensor:
        return x
    if isinstance(x, tuple):
        return tuple([to_tensor(a, Tensor) for a in x])
    if isinstance(x, list):
        return [to_tensor(a, Tensor) for a in x]
    if isinstance(x, dict):
        return {k: to_tensor(v, Tensor) for k, v in x.items()}
    if torch.is_tensor(x):
        return x.to(Tensor.dtype)
    if hasattr(x, "__iter__"):
        return Tensor(x)
    return Tensor([x]).squeeze()


def array_to_image(arr):
    im = Image.fromarray(arr.astype('uint8'), 'RGB')
    return im

def show_video(file_path):
    clip=VideoFileClip(file_path)
    return clip.ipython_display(width=280)

def vector_magnitudes(df):
    return np.sqrt((df**2).sum(axis=1))

def max_vector_magnitude(df):
    return vector_magnitudes(df).max()

def remove_leading_trailing_zeros(df, cols=y_cols):
    s = df[cols].ne(0).sum(axis=1)
    df = df[s.cumsum().ne(0) & s[::-1].cumsum().ne(0)]
    df = df.copy()
    return df

remove_zeros = remove_leading_trailing_zeros
