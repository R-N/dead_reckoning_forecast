from .datasets import MultiChannelFrameDataset
from .. import constants
import torchvision.transforms as transforms
from torchvision.utils import save_image
from ..util import mkdir
import os

def combine_frame_channels(in_dir, out_dir, channels=constants.channels, ext=".png"):
    mkdir(out_dir)
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    ds = MultiChannelFrameDataset(in_dir, channels=channels, transform=transform)
    print(len(ds))
    for i in range(len(ds)):
        img = ds[i]
        save_image(img, os.path.join(out_dir, f"{i+1}{ext}"))

def combine_frame_channels_2(data_dir, frame_dir, match_type, match_id, player):
    match_id = str(match_id)
    player = str(player)
    match_dir = os.path.join(data_dir, match_type, match_id)
    in_dir = os.path.join(match_dir, player)
    out_dir = os.path.join(frame_dir, match_type, match_id, player, "all")
    combine_frame_channels(in_dir, out_dir)
