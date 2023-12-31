import shutil
import os
import torch
from PIL import Image
import numpy as np
from collections import OrderedDict
from moviepy.editor import VideoFileClip
import pandas as pd
from pathlib import Path

Tensor = torch.Tensor
DEFAULT_DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

def stack_samples(samples):
    samples = list(zip(*samples))
    samples = [torch.stack(x) if x[0] is not None else None for x in samples]
    return samples

def to_tensor(x, Tensor=Tensor):
    if not Tensor:
        return x
    if x is None:
        return x
    if isinstance(x, tuple):
        return tuple([to_tensor(a, Tensor) for a in x])
    if isinstance(x, list):
        return [to_tensor(a, Tensor) for a in x]
    if isinstance(x, dict):
        return {k: to_tensor(v, Tensor) for k, v in x.items()}
    if isinstance(x, pd.DataFrame):
        return to_tensor(x.to_numpy())
    if isinstance(x, pd.Series):
        return to_tensor(x.to_numpy())
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

def vector_magnitudes(df, log=False):
    mag = np.sqrt((df**2).sum(axis=1))
    if log:
        mag = np.log(mag)
    return mag

def max_vector_magnitude(df, **kwargs):
    return vector_magnitudes(df, **kwargs).max()

def normalize_vector(df):
    mag = vector_magnitudes(df)
    mag = np.repeat(np.array(mag)[:, np.newaxis], 2, axis=-1)
    return df / mag

def log_vector(df):
    mag = vector_magnitudes(df, log=False)
    mul = np.log(mag) / mag
    mul = np.repeat(np.array(mul)[:, np.newaxis], 2, axis=-1)
    return df * mul

def exp_vector(df):
    mag = vector_magnitudes(df, log=False)
    mul = np.exp(mag) / mag
    mul = np.repeat(np.array(mul)[:, np.newaxis], 2, axis=-1)
    return df * mul

def remove_leading_trailing_zeros(df, cols):
    s = df[cols].ne(0).sum(axis=1)
    df = df[s.cumsum().ne(0) & s[::-1].cumsum().ne(0)]
    df = df.copy()
    return df

remove_zeros = remove_leading_trailing_zeros

class CacheType:
    MEMORY = "memory"
    PICKLE = "pickle"

    __ALL__ = (MEMORY, PICKLE)


def Cache(cache_type=CacheType.MEMORY, max_cache=None, **kwargs):
    if not max_cache:
        return None
    if cache_type == CacheType.MEMORY:
        return InMemoryCache(max_cache=max_cache, **kwargs)
    elif cache_type == CacheType.PICKLE:
        return PickleCache(max_cache=max_cache, **kwargs)

def remake_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    while os.path.exists(path):
        pass
    while True:
        try:
            mkdir(path)
            if not os.path.exists(path):
                continue
            break
        except PermissionError:
            continue

class PickleCache:
    def __init__(self, max_cache=torch.inf, remove_old=False, cache_dir="_cache", clear_first=False):
        assert cache_dir is not None
        self.remove_old = remove_old
        if clear_first:
            remake_dir(cache_dir)
        else:
            mkdir(cache_dir)
        self.cache_dir = cache_dir
        print("Caching in", cache_dir, max_cache, remove_old)

    def clear(self):
        remake_dir(self.cache_dir)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if hasattr(idx, "__iter__"):
            return stack_samples([self[id] for id in idx])
        sample = torch.load(self.file_path(idx))
        return sample
    
    def __setitem__(self, idx, sample):
        torch.save(sample, self.file_path(idx))

    def __contains__(self, idx):
        return os.path.exists(self.file_path(idx))
    
    def file_path(self, idx):
        return os.path.join(self.cache_dir, f"_cache_{idx}.pt")

class InMemoryCache:
    def __init__(self, max_cache=torch.inf, remove_old=False, cache_dir=None, clear_first=False):
        self.max_cache = max_cache
        self.cache = OrderedDict()
        self.remove_old = remove_old
        print("Caching in memory", max_cache, remove_old)

    def clear(self):
        self.cache.clear()

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if hasattr(idx, "__iter__"):
            return stack_samples([self[id] for id in idx])
        return self.cache[idx]
    
    def __setitem__(self, idx, sample):
        if len(self.cache) >= self.max_cache:
            if self.remove_old:
                self.cache.popitem(last=False)
            else:
                return
        self.cache[idx] = sample

    def __contains__(self, item):
        return item in self.cache
    
def validate_device(device="cuda"):
    return device if torch.cuda.is_available() else "cpu"


def mkdir(dir):
    Path(dir).mkdir(parents=True, exist_ok=True)

def filter_dict(dict, keys):
    return {k: v for k, v in dict.items() if k in keys}

def filter_dict_2(dict, keys):
    return {keys[k]: v for k, v in dict.items() if k in keys}

def split_df(df, points, seed=42):
    splits = np.split(
        df.sample(frac=1, random_state=seed), 
        [int(x*len(df)) for x in points]
    )
    return splits

def split_df_2(df, points, test=-1, val=None, seed=42, return_3=False):
    splits = split_df(df, points, seed=seed)

    test_df = splits[test]
    val_df = splits[val] if val else None
    train_dfs = [s for s in splits if s is not test_df and s is not val_df]
    train_df = pd.concat(train_dfs)

    if val or return_3:
        return train_df, val_df, test_df
    return train_df, test_df

def split_df_ratio(df, ratio=0.2, val=False, i=0, seed=42, return_3=False):
    count = int(1.0/ratio)
    splits = [k*ratio for k in range(1, count)]
    splits = split_df(df, splits, seed=seed)
    test_index = (count - 1 + i)%count
    val_index = (test_index-1)%count if val else None
    n = min([len(s) for s in splits])

    leftovers = []

    test_df = splits[test_index]
    leftovers.append(test_df[n:])

    val_df = test_df
    if val:
        val_df = splits[val_index]
        leftovers.append(val_df[n:])

    train_dfs = [s for s in splits if s is not test_df and s is not val_df]
    test_df = test_df[:n]
    val_df = val_df[:n]
    train_df = pd.concat(train_dfs + leftovers)

    if val or return_3:
        return train_df, val_df, test_df
    return train_df, test_df

def split_df_kfold(df, ratio=0.2, val=False, filter_i=None, seed=42, return_3=False):
    result = []
    count = int(1.0/ratio)
    splits = [k*ratio for k in range(1, count)]
    splits = split_df(df, splits, seed=seed)
    n = min([len(s) for s in splits])
    n_train = len(df) - ((2 if val else 1) * n)

    for i in range(count):
        if filter_i and i not in filter_i:
            continue
        test_index = (count - 1 + i)%count
        val_index = (test_index - 1)%count if val else None

        leftovers = []

        test_df = splits[test_index]
        leftovers.append(test_df[n:])

        val_df = test_df
        if val:
            val_df = splits[val_index]
            leftovers.append(val_df[n:])
        train_dfs = [s for s in splits if s is not test_df and s is not val_df]
        test_df = test_df[:n]
        val_df = val_df[:n]
        train_df = pd.concat(train_dfs + leftovers)

        assert len(test_df) == n, f"Invalid test length {len(test_df)} should be {n}"
        assert len(val_df) == n, f"Invalid val length {len(val_df)} should be {n}"
        assert len(train_df) == n_train, f"Invalid train length {len(train_df)} should be {n_train}"

        if val or return_3:
            result.append((train_df, val_df, test_df))
        else:
            result.append((train_df, test_df))

    return result
