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

        self.attention = get_module(attention, embed_dim=_num_features)
        self.pooling   = get_module(pooling, num_features=_num_features)

        self.frame_classifier = nn.Linear(_num_features, 2)

        if 'Statistics' in pooling['name']:
            _num_features *= 2
        
        self.utter_classifier = nn.Linear(_num_features, 2)

    def forward(self, x, mask=None):
        frontend_feature  = self.frontend(x)
        frontend_feature  = einops.rearrange(frontend_feature, 'b t f -> b () t f')


        backbone_feature  = self.backbone(frontend_feature)
        backbone_feature  = einops.rearrange(backbone_feature, 'b c t f -> b t (c f)')

        attention_feature = self.attention(backbone_feature, mask)[0]
        frame_prediction  = self.frame_classifier(attention_feature)

        pooling_feature   = self.pooling(attention_feature, mask)
        utter_prediction  = self.utter_classifier(pooling_feature)

        return frame_prediction, utter_prediction