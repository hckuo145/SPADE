import einops

import torch
import torch.nn as nn


class MultipleFrameLoss(nn.Module):
    def __init__(self):
        super(MultipleFrameLoss, self).__init__()

        weight = torch.zeros(2, 1, 17)
        weight[0] = 1.
        weight[1, 0, 8] = 1. 

        self.stride    = 10
        self.padding   = 8
        self.criterion = nn.BCELoss()
        self.register_buffer('weight', weight)

    def forward(self, input, target, mask=None):
        if mask is None:
            mask = torch.ones_like(target, dtype=torch.bool)
            
        input  = torch.softmax(input, dim=-1)[..., 1]
        
        target = target.float().masked_fill(~mask, 0.)
        target = einops.rearrange(target, 'b t -> b () t')
        target = nn.functional.conv1d(target, self.weight, stride=self.stride, padding=self.padding)[:, 0]

        output = einops.rearrange(mask.float(), 'b t -> b () t')
        output = nn.functional.conv1d(output, self.weight, stride=self.stride, padding=self.padding)
        count  = output[:, 0]
        mask   = output[:, 1] > 0.

        mask   = einops.rearrange(mask  , 'b t -> (b t)')
        input  = einops.rearrange(input , 'b t -> (b t)')
        count  = einops.rearrange(count , 'b t -> (b t)')
        target = einops.rearrange(target, 'b t -> (b t)')
        
        return self.criterion(input[mask], target[mask] / count[mask])
    

class IsolatedFrameLoss(nn.Module):
    def __init__(self, max_neighbors=3):
        super(IsolatedFrameLoss, self).__init__()

        weight = torch.zeros(max_neighbors, 1, 2 * max_neighbors + 1)
        for i in range(max_neighbors):
            s = i + 1
            weight[i, 0, max_neighbors - s:max_neighbors] = 1
            weight[i, 0, max_neighbors + 1:max_neighbors + 1 + s] = 1

        self.padding   = max_neighbors
        self.criterion = nn.L1Loss()
        self.register_buffer('weight', weight)

    def forward(self, input, mask=None):
        input  = torch.softmax(input, dim=-1)[..., 1]
        
        if mask is None:
            mask = torch.ones_like(input, dtype=torch.bool)
        
        input  = input.masked_fill(~mask, 0.)
        input  = einops.rearrange(input, 'b t -> b () t')
        target = nn.functional.conv1d(input, self.weight, padding=self.padding)
        
        mask   = einops.rearrange(mask, 'b t -> b () t')
        count  = nn.functional.conv1d(mask.float(), self.weight, padding=self.padding)

        mask   = einops.repeat(mask , 'b c t -> b (n c) t', n=target.size(1))
        input  = einops.repeat(input, 'b c t -> b (n c) t', n=target.size(1))

        mask   = einops.rearrange(mask  , 'b c t -> (b c t)')
        input  = einops.rearrange(input , 'b c t -> (b c t)')
        count  = einops.rearrange(count , 'b c t -> (b c t)')
        target = einops.rearrange(target, 'b c t -> (b c t)')

        return self.criterion(input[mask], target[mask] / count[mask])