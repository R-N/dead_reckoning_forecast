from torch import nn
import torch
import torchvision.models as models
from ..util import DEFAULT_DEVICE
from .mobile_net import mobilenet_v3_small

class MobileNet(nn.Module):
    INPUT_SIZE = (224, 224)
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    MEAN_1 = [sum(MEAN)/3]
    STD_1 = [sum(STD)/3]

    MEAN_4 = 4*MEAN_1
    STD_4 = 4*STD_1

    def __init__(self):
        super().__init__()

    @staticmethod
    def get_pretrained(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1, freeze=True, device=DEFAULT_DEVICE):
        model = models.mobilenet_v3_small(weights=weights)
        model = model.features
        for param in model.parameters():
            param.requires_grad = not freeze
        model = model.to(device)
        return model
    
    @staticmethod
    def get_custom(in_channel=4, weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1, freeze=True, device=DEFAULT_DEVICE):
        model = mobilenet_v3_small(in_channel=in_channel, weights=weights, freeze=freeze)
        model = model.features
        model = model.to(device)
        return model
