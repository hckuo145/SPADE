import einops

import torch
import torch.nn as nn

from .frontends import *
from .backbones import *
from .poolings  import *


def get_module(name_args_dict, **kwargs):
    name = name_args_dict['name']
    args = name_args_dict['args']

    return globals()[name](**args, **kwargs)


class Spade(nn.Module):
    def __init__(self, frontend, backbone, attention, pooling):
        super(Spade, self).__init__()

        _sample = torch.randn(1, 16000)

        self.frontend = get_module(frontend)

        _sample = self.frontend(_sample)
        _sample = einops.rearrange(_sample, 'b t f -> b () t f')

        self.backbone = get_module(backbone)

        _sample = self.backbone(_sample)
        _sample = einops.rearrange(_sample, 'b c t f -> b t (c f)')
        _num_features = _sample.size(-1)

        self.attention = nn.ModuleList([ get_module(attention, embed_dim=_num_features) for _ in range(2) ])
        self.pooling   = get_module(pooling, num_features=_num_features)

        self.frame_classifier = nn.Linear(_num_features, 2)

        if 'Statistics' in pooling['name']:
            _num_features *= 2
        
        self.utter_classifier = nn.Linear(_num_features, 2)

    def forward(self, x, lengths=None):
        frontend_feature = self.frontend(x)
        frontend_feature = einops.rearrange(frontend_feature, 'b t f -> b () t f')

        backbone_feature = self.backbone(frontend_feature)
        backbone_feature = einops.rearrange(backbone_feature, 'b c t f -> b t (c f)')

        mask = (torch.arange(max(lengths))[None, :] >= (lengths)[:, None]).to(x.device)
        
        attention_feature = backbone_feature
        for i in range(len(self.attention)):
            attention_feature = self.attention[i](attention_feature, mask)[0]
        frame_prediction = self.frame_classifier(attention_feature)

        pooling_feature  = self.pooling(attention_feature, mask)
        utter_prediction = self.utter_classifier(pooling_feature)

        return frame_prediction, utter_prediction
    

class Spade_v2(nn.Module):
    def __init__(self, frontend, backbone, attention):
        super(Spade_v2, self).__init__()

        _sample = torch.randn(1, 16000)

        self.frontend = get_module(frontend)

        _sample = self.frontend(_sample)
        _sample = einops.rearrange(_sample, 'b t f -> b () t f')

        self.backbone = get_module(backbone)

        _sample = self.backbone(_sample)
        _sample = einops.rearrange(_sample, 'b c t f -> b t (c f)')
        _num_features = _sample.size(-1)

        self.cls_token  = nn.Parameter(torch.zeros(_num_features))
        self.attention  = nn.ModuleList([ get_module(attention, embed_dim=_num_features) for _ in range(2) ])
        self.classifier = nn.Linear(_num_features, 2)
        
    def forward(self, x, lengths=None):
        frontend_feature = self.frontend(x)
        frontend_feature = einops.rearrange(frontend_feature, 'b t f -> b () t f')

        backbone_feature = self.backbone(frontend_feature)
        backbone_feature = einops.rearrange(backbone_feature, 'b c t f -> b t (c f)')
        
        pos_embed = self.get_pos_embed(backbone_feature.size()).to(x.device)
        cls_token = einops.repeat(self.cls_token, 'f -> b () f', b=x.size(0))
        backbone_feature = torch.cat([cls_token, backbone_feature], dim=1) + pos_embed

        attention_feature = backbone_feature
        mask = (torch.arange(max(lengths) + 1)[None, :] >= (lengths + 1)[:, None]).to(x.device)
        for i in range(len(self.attention)):
            attention_feature = self.attention[i](attention_feature, mask)[0]

        prediction = self.classifier(attention_feature)
        frame_prediction = prediction[:,1:]
        utter_prediction = prediction[:,0]

        return frame_prediction, utter_prediction
    
    @staticmethod
    def get_pos_embed(embed_size):
        position = torch.arange(embed_size[1])

        omega = torch.arange(embed_size[2] // 2, dtype=torch.float32)
        omega = 2 * omega / embed_size[2]
        omega = 1 / (1000 ** omega)

        outer = torch.einsum('t, f -> t f', position, omega)

        embed_sin = torch.sin(outer)
        embed_cos = torch.cos(outer)

        pos_embed = torch.cat([embed_sin, embed_cos], dim=1)
        pos_token = torch.zeros(1, embed_size[2])
        
        pos_embed = torch.cat([pos_token, pos_embed], dim=0)
        pos_embed = einops.repeat(pos_embed, 't f -> b t f', b=embed_size[0])

        return pos_embed