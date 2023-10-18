
import glob
from torch.utils.data import Dataset
from PIL import Image
from dead_reckoning_forecast.util import remake_dir, stack_samples, to_tensor, DEFAULT_DEVICE
import torch
from torch.utils.data import Dataset
from .. import constants
import os


class TimeSeriesDataset(Dataset):   
    def __init__(self, df, x_len=50, y_len=10, x_cols=None, y_cols=None, stride=1):
        self.df = df
        assert x_len > 0 and y_len > 0
        self.x_len = x_len
        self.y_len = y_len
        self.x_cols = x_cols or list(df.columns)
        self.y_cols = y_cols or list(df.columns)
        self.stride = stride

    def __len__(self):
        return (len(self.df) - self.y_len)//self.stride

    def __getitem__(self, index):
        if hasattr(idx, "__iter__"):
            return stack_samples([self[i] for i in idx])
        index = index * self.stride
        x = self.df.iloc[index:index+self.x_len].loc[:, self.x_cols]
        y = self.df.iloc[index+self.x_len:index+self.x_len+self.y_len].loc[:, self.y_cols]
        w = self.df.iloc[index+self.x_len:index+self.x_len+self.y_len]["weight"].copy()
        w /= list(range(1, self.y_len+1))
        return x, y, w
    

class FrameDataset(Dataset):
    def __init__(self, frame_dir, transform=None, ext=".jpg", count=0):    
        self.frame_dir = frame_dir  
        self.transform = transform
        ext = f".{ext}" if not ext.startswith(".") else ext
        print(frame_dir, ext)
        self.count = count or len(glob.glob1(frame_dir, f"*{ext}"))
        self.ext = ext

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        if hasattr(idx, "__iter__"):
            return stack_samples([self[i] for i in idx])
        
        frame_name = f"{idx+1}.jpg"
        img_path = os.path.join(self.frame_dir, frame_name)

        frame = Image.open(img_path)
        if self.transform:
            frame = self.transform(frame)
        
        #frame = to_tensor(frame, torch.Tensor)

        return frame
    
    
class MultiChannelFrameDataset(Dataset):
    def __init__(self, frame_dir, channels=constants.channels, **kwargs):
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
        channels = [d[idx] for d in self.dataset_list]
        frame = torch.cat(channels, dim=0)
        
        if self.transform:
            frame = self.transform(frame)

        return frame
        