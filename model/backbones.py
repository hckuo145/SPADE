import einops

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .sublayers import *


class ResNet(nn.Module):
    def __init__(self, block_fn='SEBasicBlock', blocks=[3, 4, 6, 3], channels=[16, 16, 32, 64, 128], sqex_reduce=4, btnk_reduce=1, scale=None, dropout=0.):
        super(ResNet, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, channels[0], kernel_size=7, stride=(1, 2), padding=3, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))

        block_fn    = globals()[block_fn]
        block_args  = {'sqex_reduce': sqex_reduce, 'btnk_reduce': btnk_reduce, 'scale': scale, 'dropout': dropout}
        self.layer1 = self._make_layer(block_fn, blocks[0], channels[0], channels[1], **block_args)
        self.layer2 = self._make_layer(block_fn, blocks[1], channels[1], channels[2], **block_args)
        self.layer3 = self._make_layer(block_fn, blocks[2], channels[2], channels[3], **block_args)
        self.layer4 = self._make_layer(block_fn, blocks[3], channels[3], channels[4], **block_args)
    
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block_fn, num_blocks, in_channels, out_channels, stride=1, **kwargs):
        layer = []
        layer.append(block_fn(in_channels, out_channels, stride, **kwargs))
        for _ in range(1, num_blocks):
            layer.append(block_fn(out_channels, out_channels, 1, **kwargs))

        return nn.Sequential(*layer)

    def forward(self, x):
        out = self.conv(x)
        out = self.pool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        return out
    

class ResConv1d_BLSTM(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.):
        super(ResConv1d_BLSTM, self).__init__()

        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels if i == 0 else out_channels, out_channels, \
                        kernel_size=3, padding=1, bias=False),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
            ) for i in range(7)
        ])
        # self.drop = nn.Dropout(dropout)

        self.lstm = nn.LSTM(out_channels, out_channels // 2, bidirectional=True, batch_first=True)

        self.adpt = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels)
        ) if in_channels != out_channels else None

    def forward(self, x, lengths=None):
        conv_out = x
        for i, conv in enumerate(self.convs):
            conv_res = conv(conv_out)
            # conv_res = self.drop(conv_res)
            if i == 0 and self.adpt is not None:
                conv_out = self.adpt(conv_out)
            conv_out = conv_out + conv_res

        lstm_out = einops.rearrange(conv_out, 'b f t -> b t f')
        if lengths is None:
            lstm_out = self.lstm(lstm_out)[0]
        else:
            lstm_out = pack_padded_sequence(lstm_out, lengths=lengths, batch_first=True, enforce_sorted=False)
            lstm_out = self.lstm(lstm_out)[0]
            lstm_out = pad_packed_sequence(lstm_out, batch_first=True)[0]

        return conv_out, lstm_out
    

class Conv2d_BGRU(nn.Module):
    def __init__(self, channels=[32, 64, 128, 128, 128], dropout=0.):
        super(Conv2d_BGRU, self).__init__()
        
        conv = []
        in_channels = 1
        for out_channels in channels:
            conv += [
                Conv3x3(in_channels, out_channels),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),

                Conv3x3(out_channels, out_channels),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),

                nn.AvgPool2d(kernel_size=(1, 2), stride=(1,2)),
                # nn.Dropout(dropout, inplace=True)
            ]
            
            in_channels = out_channels
        self.conv = nn.Sequential(*conv)
        
        self.gru  = nn.GRU(out_channels, out_channels // 2, num_layers=2, dropout=dropout, bidirectional=True, batch_first=True)

    def forward(self, x, lengths=None):
        out = self.conv(x)

        out = torch.mean(out, dim=3)
        out = einops.rearrange(out, 'b c t -> b t c')

        if lengths is None:
            out = self.gru(out)[0]
        else:
            out = pack_padded_sequence(out, lengths=lengths, batch_first=True, enforce_sorted=False)
            out = self.gru(out)[0]
            out = pad_packed_sequence(out, batch_first=True)[0]

        return out