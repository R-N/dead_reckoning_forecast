from torch import nn
import torch
import torch.functional as F
import math
from ..util import DEFAULT_DEVICE

class CRNN(nn.Module):
    def __init__(
        self, 
        cnn, 
        d_state=12,
        d_out=2, 
        d_cnn_0=576,
        d_cnn=64, 
        d_adapt=16,
        d_rnn=256,
        horizon=10,
        RNN=nn.GRU, 
        activation=nn.ReLU, 
        size_cnn=(7, 7), 
        activation_final=nn.Tanh,
        device=DEFAULT_DEVICE
    ):
        super().__init__()
        self.cnn = cnn
        self.do = nn.Dropout(0.2)
        self.activation = activation()
        self.activation_final = activation_final()
        
        #d_cnn_1 = d_cnn-960
        #"""
        self.cnn_1 = nn.Sequential(
            nn.Conv2d(d_cnn_0, d_cnn, 3, padding=1),
            self.activation,
            nn.BatchNorm2d(d_cnn),
            self.do,
        )
        """
        self.cnn_2 = nn.Sequential(
            nn.Conv2d(960, 960, 1),
            self.activation,
            nn.BatchNorm2d(960),
            self.do,
        )
        """
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        """
        self.adapter = nn.Sequential(
            nn.Linear(d_cnn, d_rnn),
            self.activation,
            self.do
        )
        """
        self.d_state = d_state
        self.d_adapt = d_adapt
        self.adapter_0 = nn.Sequential(
            nn.Linear(d_state, d_adapt),
            self.activation,
        )
        self.adapter_i = nn.Sequential(
            nn.Linear(d_adapt, d_adapt),
            self.activation,
        )
        self.adapter_n = nn.Sequential(
            nn.Linear(d_cnn+d_adapt, d_rnn),
            self.activation,
        )
        self.rnn = RNN(input_size=d_rnn, hidden_size=d_rnn, batch_first=True)
        self.final_i = nn.Sequential(
            nn.Linear(d_rnn, d_rnn),
            self.activation
        )
        self.final_n = nn.Sequential(
            nn.Linear(d_rnn, d_out),
            self.activation_final,
        )
        #self.take = torch.LongTensor([-1])
        self.horizon=horizon
        self.device = device
        self.to(device)
        
    def forward(self, x, frames):

        b = frames.shape[-5] if frames.dim() > 4 else None
        l = frames.shape[-4]
        cnn_shape = (-1, *frames.shape[-3:])
        frames = frames.view(cnn_shape)
        frames = self.cnn(frames)
        frames = frames_1 = self.cnn_1(frames)
        #frames_2 = self.cnn_2(frames)
        #frames = torch.cat([frames_1,frames_2], dim=-3)
        frames = self.pool(frames)
        pre_rnn_shape = (l, -1)
        if b:
            pre_rnn_shape = (b, *pre_rnn_shape)
        frames = frames.view(pre_rnn_shape)

        x = self.adapter_0(x)
        x = x + self.adapter_i(x)

        x = torch.cat([x, frames], dim=-1)
        x = self.adapter_n(x)
    
        x, h = self.rnn(x)

        x, h = take_last(x, b), take_last(x, h)

        preds = []
        for i in range(self.horizon):
            x, h = self.rnn(x, h)
            x, h = take_last(x, b), take_last(x, h)
            x = x + self.final_i(x)
            x = self.final_n(x)
            preds.append(x)

        pred = torch.stack(preds)
        return pred
        
def take_last(x, b):
    x = x[:, -1:, :] if b else x[-1:, :]
    return x
