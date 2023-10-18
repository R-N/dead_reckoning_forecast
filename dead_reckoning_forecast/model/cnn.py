from torch import nn
import torchvision.models as models
from ..util import DEFAULT_DEVICE

class MobileNet(nn.Module):
    INPUT_SIZE = (224, 224)
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]

    def __init__(self):
        super().__init__()

    @staticmethod
    def get_pretrained(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2, freeze=True, device=DEFAULT_DEVICE):
        model = models.mobilenet_v3_small(weights=weights)
        model = list(model.children())[0]
        for param in model.parameters():
            param.requires_grad = not freeze
        model = model.to(device)
        return model
