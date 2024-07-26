import torch
import torch.nn as nn


def Conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


def Conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


class SqueezeAndExcite(nn.Module):
    def __init__(self, num_channels, reduce=16):
        super(SqueezeAndExcite, self).__init__()

        rdc_channels = int(num_channels / reduce)

        self.excite = nn.Sequential(
            nn.Linear(num_channels, rdc_channels, bias=False),
            nn.ReLU(inplace=True),

            nn.Linear(rdc_channels, num_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        s = torch.mean(x, dim=(2, 3))
        e = self.excite(s)
        y = torch.einsum('b c h w, b c -> b c h w', x, e)

        return y

class SEBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, sqex_reduce=16, btnk_reduce=1, dropout=0., **kwargs):
        super(SEBasicBlock, self).__init__()

        rdc_channels = int(out_channels / btnk_reduce)

        self.conv = nn.Sequential(
            Conv3x3(in_channels, rdc_channels, stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            Conv3x3(rdc_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )
 
        self.sqex = SqueezeAndExcite(out_channels, sqex_reduce)
        # self.drop = nn.Dropout(dropout, inplace=True)
        self.relu = nn.ReLU(inplace=True)
        
        self.adpt = nn.Sequential(
            Conv1x1(in_channels, out_channels, stride),
            nn.BatchNorm2d(out_channels)
        ) if stride != 1 or in_channels != out_channels else None

    def forward(self, x):
        res = self.conv(x)
        res = self.sqex(res)
        # res = self.drop(res)

        if self.adpt is not None:
            x = self.adpt(x)

        out = x + res
        out = self.relu(out)

        return out


class SEBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, sqex_reduce=16, btnk_reduce=4, dropout=0., **kwargs):
        super(SEBottleneck, self).__init__()

        rdc_channels = int(out_channels / btnk_reduce)

        self.conv = nn.Sequential(
            Conv1x1(in_channels, rdc_channels),
            nn.BatchNorm2d(rdc_channels),
            nn.ReLU(inplace=True),

            Conv3x3(rdc_channels, rdc_channels, stride),
            nn.BatchNorm2d(rdc_channels),
            nn.ReLU(inplace=True),

            Conv1x1(rdc_channels, out_channels),
            nn.BatchNorm2d(out_channels)
        )

        self.sqex = SqueezeAndExcite(out_channels, sqex_reduce)
        # self.drop = nn.Dropout(dropout, inplace=True)
        self.relu = nn.ReLU(inplace=True)

        self.adpt = nn.Sequential(
            Conv1x1(in_channels, out_channels, stride),
            nn.BatchNorm2d(out_channels)
        ) if stride != 1 or in_channels != out_channels else None
        
    def forward(self, x):
        res = self.conv(x)
        res = self.sqex(res)
        # res = self.drop(res)

        if self.adpt is not None:
            x = self.adpt(x)

        out = x + res
        out = self.relu(out)

        return out


class SEBottle2neck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, sqex_reduce=16, btnk_reduce=4, dropout=0., scale=4):
        super(SEBottle2neck, self).__init__()

        rdc_channels = int(out_channels / btnk_reduce)

        self.conv1 = nn.Sequential(
            Conv1x1(in_channels, rdc_channels * scale),
            nn.BatchNorm2d(rdc_channels * scale),
            nn.ReLU(inplace=True)
        )

        self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1) \
                if stride != 1 or in_channels != out_channels else None

        self.convs = nn.ModuleList([ 
            nn.Sequential(
                Conv3x3(rdc_channels, rdc_channels, stride),
                nn.BatchNorm2d(rdc_channels),
                nn.ReLU(inplace=True)
            ) for _ in range(scale - 1) 
        ])

        self.conv3 = nn.Sequential(
            Conv1x1(rdc_channels * scale, out_channels),
            nn.BatchNorm2d(out_channels)
        )
        
        self.sqex = SqueezeAndExcite(out_channels, sqex_reduce)
        # self.drop = nn.Dropout(dropout, inplace=True)
        self.relu = nn.ReLU(inplace=True)
    
        self.adpt = nn.Sequential(
            Conv1x1(in_channels, out_channels, stride),
            nn.BatchNorm2d(out_channels)
        ) if stride != 1 or in_channels != out_channels else None

        self.scale = scale

    def forward(self, x):
        res = self.conv1(x)

        sps = torch.chunk(res, self.scale, dim=1)
        
        res = sps[0]
        if self.adpt is not None:
            res = self.pool(res)
        
        for i, conv in enumerate(1, self.convs):
            if i == 1 or self.adpt is not None:
                spx = sps[i]
            else:
                spx = spx + sps[i]

            spx = conv(spx)
            res = torch.cat([res, spx], dim=1)

        res = self.conv3(res)
        res = self.sqex(res)
        # res = self.drop(res)

        if self.adpt is not None:
            x = self.adpt(x)

        out = x + res
        out = self.relu(out)

        return out
        

class Transformer(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim, dropout=0.):
        super(Transformer, self).__init__()

        self.attn  = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.drop1 = nn.Dropout(dropout, inplace=True)
        self.norm1 = nn.LayerNorm(embed_dim, eps=1e-9)

        self.ffn   = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embed_dim)
        )
        self.drop2 = nn.Dropout(dropout, inplace=True)
        self.norm2 = nn.LayerNorm(embed_dim, eps=1e-9)

    def forward(self, x, mask=None):
        res, mat = self.attn(x, x, x, mask)
        res = self.drop1(res)
        out = self.norm1(x + res)

        res = self.ffn(out)
        res = self.drop2(res)
        out = self.norm2(out + res)

        return out, mat