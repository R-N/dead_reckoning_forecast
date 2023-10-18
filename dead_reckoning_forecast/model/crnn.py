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
        activation_adapter_final=nn.Tanh,
        activation_final=nn.Tanh,
        device=DEFAULT_DEVICE
    ):
        super().__init__()
        self.cnn = cnn
        self.do = nn.Dropout(0.2)
        self.activation = activation()
        self.activation_adapter_final = activation_adapter_final()
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
            self.activation_adapter_final,
        )
        self.rnn = RNN(input_size=d_rnn, hidden_size=d_rnn, batch_first=True)
        self.final_0 = nn.Sequential(
            nn.Linear(d_rnn, d_rnn),
            self.activation
        )
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
    
        x_0 = x[:, 1:, :] if b else x[1:, :]

        xy = None

        if True:
            xy = x
            xy = xy + self.final_0(xy)
            xy = xy + self.final_i(xy)
            xy = self.final_n(xy)

        x, h = self.rnn(x)
        x = x + self.final_0(x)

        x_1 = x[:, :-1, :] if b else x[:-1, :]
        
        if self.training:
            x = x[:, -self.horizon:, :] if b else x[-self.horizon:, :]
        else:
            x, h = take_last_rnn(x, h, b)
            preds = []
            preds.append(x.squeeze(-2))
            for i in range(self.horizon-1):
                x, h = self.rnn(x, h)
                x, h = take_last_rnn(x, h, b)
                x = x + self.final_0(x)
                preds.append(x.squeeze(-2))

            x = torch.stack(preds, dim=-2)

        x = x + self.final_i(x)
        x = self.final_n(x)

        return x, (x_0, x_1), xy
    
def take_last_rnn(x, h, b):
    x = take_last(x, b)
    h = permute_batch(h, b)
    h = take_last(h, b)
    h = permute_batch(h, b)
    return x, h

    
def permute_batch(x, b):
    if isinstance(x, tuple):
        return tuple([permute_batch(xi, b) for xi in x])
    if b:
        return x.permute(-2, -3, -1)
    return x
        
def take_last(x, b):
    if isinstance(x, tuple):
        return  tuple([take_last(xi, b) for xi in x])
    x = x[:, -1:, :] if b else x[-1:, :]
    return x
