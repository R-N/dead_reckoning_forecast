from torch.utils.data import Dataset, DataLoader, Subset
import glob
from PIL import Image
import torch
import numpy as np
import random
from torchvision.io import read_image
from ..util import stack_samples, to_tensor

#np.random.seed(42)
#random.seed(42)
#torch.manual_seed(42)

class VideoDataset(Dataset):
    def __init__(self, info, frame_dir, transform=None, n_frames=16, label_encoder=None):      
        self.transform = transform
        self.info = info
        self.n_frames = n_frames
        self.frame_dir = frame_dir
        self.label_encoder=label_encoder

    def __len__(self):
        return len(self.info)

    def __getitem__(self, idx):
        if hasattr(idx, "__iter__"):
            return stack_samples([self[i] for i in idx])
        file_info = self.info[idx]
        file_path = file_info["file_path"]
        frame_path = file_info["frame_path"]
        frame_names = [f"frame_{i}.jpg" for i in range(self.n_frames)]
        img_paths = [os.path.join(frame_path, fn) for fn in frame_names]
        
        if "label" in file_info:
            label = file_info["label"]
        elif self.label_encoder:
            label = np.array([file_info["tag"]])
            label = self.label_encoder.transform(label)
        else:
            raise RuntimeError("Either label or label_encoder must be provided")

        frames = [Image.open(i) for i in img_paths]
        frames = [self.transform(f) for f in frames]
        
        frames = torch.stack([f for f in frames])
        label = to_tensor(label, torch.LongTensor)

        return frames, label