
import glob
from torch.utils.data import Dataset
from PIL import Image
from dead_reckoning_forecast.util import remake_dir, stack_samples, to_tensor, DEFAULT_DEVICE
import torch
from torch.utils.data import Dataset
from .. import constants
from ..util import Cache, split_df_ratio, split_df_kfold
import os
import pandas as pd
import numpy as np
import gc

class BaseDataset(Dataset):
    def __init__(self, max_cache=None, **cache_kwargs):
        self.create_cache(max_cache, **cache_kwargs)

    def create_cache(self, max_cache, **cache_kwargs):
        if hasattr(self, "max_cache") and self.max_cache == max_cache:
            return
        if max_cache == True:
            max_cache = torch.inf
        self.max_cache = max_cache
        self.cache = Cache(max_cache, **cache_kwargs) if max_cache else None

    @property
    def index(self):
        return list(range(len(self)))

    def clear_cache(self):
        if self.cache:
            self.cache.clear()
            gc.collect()

    def slice(self, start=0, stop=None, step=1):
        index = pd.Series(self.index)
        stop = stop or (len(index)-1)
        sample = index[start:stop:step]
        dataset = SubDataset(self, sample)
        return dataset

    def sample(self, **kwargs):
        index = pd.Series(self.index)
        sample = index.sample(**kwargs)
        dataset = SubDataset(self, sample)
        return dataset

    def split_ratio(self, **kwargs):
        index = pd.Series(self.index)
        datasets = split_df_ratio(index, **kwargs)
        datasets = [SubDataset(self, i) for i in datasets]
        return datasets

    def split_kfold(self, **kwargs):
        index = pd.Series(self.index)
        splits = split_df_kfold(index, **kwargs)
        splits = [[SubDataset(self, i) for i in datasets] for datasets in splits]
        return splits

class WrapperDataset(BaseDataset):
    def __init__(self, dataset, max_cache=None):
        super().__init__(max_cache=max_cache)
        self.dataset = dataset

    @property
    def index(self):
        return self.dataset.index

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]


class SubDataset(WrapperDataset):
    def __init__(self, dataset, index, max_cache=None):
        super().__init__(dataset=dataset, max_cache=max_cache)
        if isinstance(index, pd.Series) or isinstance(index, pd.Index):
            index = index.to_numpy()
        if not isinstance(index, np.ndarray):
            index = np.array(index)
        index = index.astype(int)
        self.index_ = index

    @property
    def index(self):
        return self.index_

    def __len__(self):
        return len(self.index_)

    def __getitem__(self, idx):
        return self.dataset[self.index[idx]]

class TimeSeriesDataset(BaseDataset):   
    def __init__(self, df, x_len=50, y_len=10, x_cols=None, y_cols=None, stride=1, max_cache=None):
        super().__init__(max_cache=max_cache)
        self.df = df
        assert x_len > 0 and y_len > 0
        self.x_len = x_len
        self.y_len = y_len
        self.x_cols = x_cols or list(df.columns)
        self.y_cols = y_cols or list(df.columns)
        self.stride = stride

    def __len__(self):
        return (len(self.df) - self.y_len)//self.stride

    def __getitem__(self, idx):
        if hasattr(idx, "__iter__"):
            return stack_samples([self[i] for i in idx])
        
        if self.cache and idx in self.cache:
            return self.cache[idx]
    
        idx = idx * self.stride
        x = self.df.iloc[idx:idx+self.x_len].loc[:, self.x_cols]
        y = self.df.iloc[idx+self.x_len:idx+self.x_len+self.y_len].loc[:, self.y_cols]
        w = self.df.iloc[idx+self.x_len:idx+self.x_len+self.y_len]["weight"].copy()
        w /= list(range(1, self.y_len+1))

        sample = x, y, w

        if self.cache:
            self.cache[idx] = sample

        return sample
    

class FrameDataset(BaseDataset):
    def __init__(self, frame_dir, transform=None, ext=".jpg", count=0, max_cache=None):
        super().__init__(max_cache=max_cache)
        self.frame_dir = frame_dir  
        self.transform = transform
        ext = f".{ext}" if not ext.startswith(".") else ext
        self.count = count or len(glob.glob1(frame_dir, f"*{ext}"))
        self.ext = ext

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        if hasattr(idx, "__iter__"):
            return stack_samples([self[i] for i in idx])
        
        if self.cache and idx in self.cache:
            return self.cache[idx]
        
        frame_name = f"{idx+1}{self.ext}"
        img_path = os.path.join(self.frame_dir, frame_name)

        frame = Image.open(img_path)
        if self.transform:
            frame = self.transform(frame)
        
        #frame = to_tensor(frame, torch.Tensor)

        if self.cache:
            self.cache[idx] = frame

        return frame
    
    
class MultiChannelFrameDataset(BaseDataset):
    def __init__(self, frame_dir, channels=constants.channels, max_cache=None, **kwargs):
        super().__init__(max_cache=max_cache)
        self.channels = channels
        self.transform = None
        self.dataset_dict = {
            c: FrameDataset(os.path.join(frame_dir, c), **kwargs)
            for c in channels
        }
        self.dataset_list = [self.dataset_dict[c] for c in channels]
        counts = [len(d) for d in self.dataset_list]
        assert len(set(counts)) == 1, f"Counts not the same, {counts}"
        self.count = counts[0]

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        if hasattr(idx, "__iter__"):
            return stack_samples([self[i] for i in idx])
        
        if self.cache and idx in self.cache:
            return self.cache[idx]

        channels = [d[idx] for d in self.dataset_list]
        frame = torch.cat(channels, dim=0)
        
        if self.transform:
            frame = self.transform(frame)

        if self.cache:
            self.cache[idx] = frame

        return frame
        