
import glob
from torch.utils.data import Dataset
from PIL import Image
import torch
from torch.utils.data import Dataset
from .. import constants
from ..util import Cache, split_df_ratio, split_df_kfold, to_tensor, stack_samples
import os
import pandas as pd
import numpy as np
import gc

class BaseDataset(Dataset):
    def __init__(self, max_cache=None, val=None, **kwargs):
        self.val = val
        self.create_cache(max_cache=max_cache, **kwargs)

    def create_cache(self, max_cache=None, cache_dir="_cache", force=False, **kwargs):
        if not force and hasattr(self, "max_cache") and self.max_cache == max_cache:
            return
        if max_cache is None and hasattr(self, "max_cache"):
            max_cache == self.max_cache
        if max_cache == True:
            max_cache = torch.inf
        self.max_cache = max_cache

        self.cache = Cache(max_cache=max_cache, cache_dir=cache_dir, **kwargs) if max_cache else None

    @property
    def index(self):
        return list(range(len(self)))

    def clear_cache(self):
        if self.cache:
            self.create_cache(force=True)
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
    
    def get(self, idx, val=None):
        return self[idx]
    

class WrapperDataset(BaseDataset):
    def __init__(self, dataset, max_cache=None, val=None, **kwargs):
        super().__init__(max_cache=max_cache, val=val, **kwargs)
        self.dataset = dataset

    @property
    def index(self):
        return self.dataset.index

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.get(idx, val=None)
    
    def get(self, idx, val=None):
        val = self.val if val is None else val
        return self.dataset.get(idx, val=val)


class SubDataset(WrapperDataset):
    def __init__(self, dataset, index, max_cache=None, val=None, **kwargs):
        super().__init__(dataset=dataset, max_cache=max_cache, val=val, **kwargs)
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
        return self.get(idx, val=None)
    
    def get(self, idx, val=None):
        val = self.val if val is None else val
        return self.dataset.get(self.index[idx], val=val)

class TimeSeriesDataset(BaseDataset):   
    def __init__(self, df, x_len=50, y_len=10, x_cols=None, y_cols=None, stride=1, max_cache=None, val=None, **kwargs):
        super().__init__(max_cache=max_cache, val=val, **kwargs)
        self.df = df
        assert x_len > 0 and y_len > 0
        self.x_len = x_len
        self.y_len = y_len
        self.x_cols = x_cols or list(df.columns)
        self.y_cols = y_cols or list(df.columns)
        self.stride = stride

    @property
    def w(self):
        h = self.y_len
        return np.array([(h-i)/h for i in range(h)])

    def __len__(self):
        return (len(self.df) - self.x_len - self.y_len + 1)//self.stride

    def __getitem__(self, idx):
        return self.get(idx)

    def get(self, idx, val=None):
        if hasattr(idx, "__iter__"):
            return stack_samples([self[i] for i in idx])
        
        if self.cache and idx in self.cache:
            return self.cache[idx]
    
        idx = idx * self.stride
        val = self.val if val is None else val

        x_stop = (idx+self.x_len) if val else (idx+self.x_len+self.y_len-1)
        x = self.df.iloc[idx:x_stop].loc[:, self.x_cols]
        y = self.df.iloc[idx+self.x_len:idx+self.x_len+self.y_len].loc[:, self.y_cols]
        w = self.df.iloc[idx+self.x_len:idx+self.x_len+self.y_len]["weight"].copy()
        w *= self.w
        #xy = None if val else self.df.iloc[idx:x_stop].loc[:, self.y_cols]
        xy = self.df.iloc[idx:x_stop].loc[:, self.y_cols]

        sample = x, y, w, xy

        if self.cache:
            self.cache[idx] = sample

        return sample
    
    def total_y_weight(self, idx):
        if hasattr(idx, "__iter__"):
            return [self.total_y_weight(i) for i in idx]
        idx = idx * self.stride
        x_stop = (idx+self.x_len) 
        yw = self.df.iloc[idx+self.x_len:idx+self.x_len+self.y_len]["weight"]
        yw = yw.sum()
        return yw
    
    @property
    def non_constant_y_index(self):
        idx = self.index
        weights = self.total_y_weight(idx)
        idx = [i for i, w in zip(idx, weights) if w > self.y_len]
        return idx
    
    def total_weight(self, idx):
        if hasattr(idx, "__iter__"):
            return [self.total_weight(i) for i in idx]
        idx = idx * self.stride
        x_stop = (idx+self.x_len) 
        xw = self.df.iloc[idx:x_stop]["weight"]
        yw = self.df.iloc[idx+self.x_len:idx+self.x_len+self.y_len]["weight"]
        w = xw.sum() + yw.sum()
        return w
    
    @property
    def non_constant_index(self):
        idx = self.index
        weights = self.total_weight(idx)
        idx = [i for i, w in zip(idx, weights) if w > (self.x_len + self.y_len)]
        return idx

    @property
    def non_constant_y_subset(self):
        index = pd.Series(self.non_constant_y_index)
        dataset = SubDataset(self, index)
        return dataset

    @property
    def non_constant_subset(self):
        index = pd.Series(self.non_constant_index)
        dataset = SubDataset(self, index)
        return dataset
    

class FrameDataset(BaseDataset):
    def __init__(self, frame_dir, transform=None, ext=".jpg", count=0, max_cache=None, val=None, **kwargs):
        super().__init__(max_cache=max_cache, val=val, **kwargs)
        self.frame_dir = frame_dir  
        self.transform = transform
        ext = f".{ext}" if not ext.startswith(".") else ext
        self.count = count or len(glob.glob1(frame_dir, f"*{ext}"))
        self.ext = ext

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        if hasattr(idx, "__iter__"):
            return torch.stack([self[i] for i in idx])
        
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
    def __init__(self, frame_dir, channels=constants.channels, max_cache=None, val=None, **kwargs):
        super().__init__(max_cache=max_cache, val=val)
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
            return torch.stack([self[i] for i in idx])
        
        if self.cache and idx in self.cache:
            return self.cache[idx]

        channels = [d[idx] for d in self.dataset_list]
        frame = torch.cat(channels, dim=0)
        
        if self.transform:
            frame = self.transform(frame)

        if self.cache:
            self.cache[idx] = frame

        return frame
        
        
class TimeSeriesFrameDataset(BaseDataset):   
    def __init__(self, ts_dataset, frame_dataset, max_cache=None, val=None, **kwargs):
        super().__init__(max_cache=max_cache, val=val, **kwargs)
        self.ts_dataset = ts_dataset
        self.frame_dataset = frame_dataset

    def __len__(self):
        return len(self.ts_dataset)
    
    def __getitem__(self, idx):
        return self.get(idx)

    def get(self, idx, val=None):
        if hasattr(idx, "__iter__"):
            return torch.stack([self[i] for i in idx])
        
        if self.cache and idx in self.cache:
            return self.cache[idx]

        val = self.val if val is None else val

        x, y, w, xy = self.ts_dataset.get(idx, val=val)
        frames = self.frame_dataset.get(x.index, val=val)
    
        x = to_tensor(x)
        y = to_tensor(y)
        w = to_tensor(w)
        xy = to_tensor(xy)
        
        sample =  x, frames, y, w, xy

        if self.cache:
            self.cache[idx] = sample

        return sample

def create_ts_frame_dataset(ts_dataset, frame_dir, match_type, match_id, player, transformer=None, frame_kwargs={}, **kwargs):
    match_id = str(match_id)
    player = str(player)
    combined_frame_dir = os.path.join(frame_dir, match_type, match_id, player, "all")
    frames = FrameDataset(combined_frame_dir, transform=transformer, ext=".png", **frame_kwargs)
    dataset = TimeSeriesFrameDataset(ts_dataset, frames, **kwargs)
    return dataset
    