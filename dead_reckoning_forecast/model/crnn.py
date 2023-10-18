from torch import nn
import torch
import torch.functional as F
import math

class CRNN(nn.Module):
    def __init__(self, cnn, n_classes, d_cnn=1024, d_rnn=256, RNN=nn.GRU, activation=nn.ReLU, size_cnn=(7, 7)):
        super().__init__()
        self.cnn = cnn
        self.do = nn.Dropout(0.2)
        self.activation = activation()
        
        #d_cnn_1 = d_cnn-960
        #"""
        self.cnn_1 = nn.Sequential(
            nn.Conv2d(960, d_cnn, 3, padding=1),
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
        self.rnn = RNN(input_size=d_cnn, hidden_size=d_rnn, batch_first=False)
        self.final = nn.Sequential(
            nn.Linear(d_rnn, d_rnn),
            self.activation,
            nn.Linear(d_rnn, n_classes),
            nn.Softmax(),
        )
        self.take = torch.LongTensor([-1])
        
    def forward(self, x):
        b = x.shape[-5] if x.dim() > 4 else None
        l = x.shape[-4]
        cnn_shape = (-1, *x.shape[-3:])
        x = x.view(cnn_shape)
        x = self.cnn(x)
        x = x_1 = self.cnn_1(x)
        #x_2 = self.cnn_2(x)
        #x = torch.cat([x_1, x_2], dim=-3)
        x = self.pool(x)
        pre_rnn_shape = (l, -1)
        if b:
            pre_rnn_shape = (b, *pre_rnn_shape)
        x = x.view(pre_rnn_shape)
        #x = self.adapter(x)
        x, h = self.rnn(x)
        x = x[:, -1, :] if b else x[-1, :]
        x = self.final(x)
        return x
        