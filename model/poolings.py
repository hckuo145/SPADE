import einops

import torch
import torch.nn as nn


class TemporalAveragePooling(nn.Module):
    def __init__(self, **kwargs):
        super(TemporalAveragePooling, self).__init__()

    def forward(self, x, mask=None):
        if mask is None:
            x = torch.mean(x, dim=1)
        else:
            mask  = einops.repeat(mask, 'b t -> b t f', f=x.size(2))
            count = torch.sum(~mask, dim=1).type(torch.float32).clamp(min=1e-12)

            x_msk = x.masked_fill(mask, 0.)
            x_sum = torch.sum(x_msk, dim=1)

            x = x_sum / count

        return x


class TemporalStatisticsPooling(nn.Module):
    def __init__(self, **kwargs):
        super(TemporalStatisticsPooling, self).__init__()

    def forward(self, x, mask=None):
        if mask is None:
            mu = torch.mean(x, dim=1)
            rh = torch.std(x, dim=1)
            x  = torch.cat([mu, rh], dim=1)
        else:
            mask  = einops.repeat(mask, 'b t -> b t f', f=x.size(2))
            count = torch.sum(~mask, dim=1).type(torch.float32).clamp(min=1e-12)

            x_msk = x.masked_fill(mask, 0.)
            x_sum = torch.sum(x_msk, dim=1)
            mu    = x_sum / count

            xx_msk = (x ** 2).masked_fill(mask, 0.)
            xx_sum = torch.sum(xx_msk, dim=1)
            rh     = torch.sqrt((xx_sum / count - mu ** 2).clamp(min=1e-12))

            x = torch.cat([mu, rh], dim=1)

        return x


class SelfAttentivePooling(nn.Module):
    def __init__(self, num_features):
        super(SelfAttentivePooling, self).__init__()

        self.proj = nn.Linear(num_features, num_features)
        self.attn = nn.Parameter(torch.ones(num_features) / num_features)

    def forward(self, x, mask=None):
        h = torch.tanh(self.proj(x))
        w = torch.einsum('b t f, f -> b t', h, self.attn)
        
        if mask is not None:
            w = w.masked_fill(mask, torch.finfo(torch.float32).min)
        w = torch.softmax(w, dim=1)     

        x = torch.einsum('b t f, b t -> b t f', x, w)
        x = torch.sum(x, dim=1) 

        return x
    

class AttentiveStatisticsPooling(nn.Module):
    def __init__(self, num_features):
        super(AttentiveStatisticsPooling, self).__init__()

        self.proj = nn.Linear(num_features, num_features)
        self.attn = nn.Parameter(torch.ones(num_features) / num_features)

    def forward(self, x, mask=None):
        h = torch.tanh(self.proj(x))
        w = torch.einsum('b t f, f -> b t', h, self.attn)
        w = torch.softmax(w, dim=1)
        
        if mask is not None:
            w = w.masked_fill(mask, torch.finfo(torch.float32).min)
        w = torch.softmax(w, dim=1)     

        mu = torch.einsum('b t f, b t -> b t f', x, w)
        mu = torch.sum(mu, dim=1)

        rh = torch.einsum('b t f, b t -> b t f', x ** 2, w)
        rh = torch.sum(rh, dim=1)
        rh = torch.sqrt((rh - mu ** 2).clamp(min=1e-12))

        x = torch.cat([mu, rh], dim=1)

        return x