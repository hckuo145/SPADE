import torch
import torch.nn as nn

from .sublayers import *


class ResNet(nn.Module):
    def __init__(self, block_fn='SEBasicBlock', blocks=[3, 4, 6, 3], channels=[16, 16, 32, 64, 128], sqex_reduce=4, btnk_reduce=1, scale=None):
        super(ResNet, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, channels[0], kernel_size=7, stride=(1, 2), padding=3, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 2), padding=(0, 1))

        block_fn    = globals()[block_fn]
        block_args  = {'sqex_reduce': sqex_reduce, 'btnk_reduce': btnk_reduce, 'scale': scale}
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