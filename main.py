import yaml
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model   import *
from runner  import Runner
from dataset import HalfTruthDataset


parser = argparse.ArgumentParser()
parser.add_argument('--test' , action='store_true', default=False)
parser.add_argument('--train', action='store_true', default=False)

parser.add_argument('--seed'  , type=int, default=0)
parser.add_argument('--batch' , type=int, default=256)
parser.add_argument('--model' , type=str, default='Spade')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--params', type=str, default=None)

parser.add_argument('--title'     , type=str, default='untitled')
parser.add_argument('--exp_path'  , type=str, default='exp/')
parser.add_argument('--model_conf', type=str, default='config/model/model.yaml')
parser.add_argument('--hyper_conf', type=str, default='config/hyper/hyper.yaml')
args = parser.parse_args()

with open(args.hyper_conf) as conf:
    vars(args).update(yaml.load(conf, Loader=yaml.Loader))

with open(args.model_conf) as conf:
    model_args = yaml.load(conf, Loader=yaml.Loader)


np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
device = torch.device(args.device)

print(f'[Title] - Task name: {args.title}', flush=True)

model = globals()[args.model](**model_args).to(device)
num_params = sum( params.numel() for params in model.parameters() )
print(f'[Model] - # params: {num_params}', flush=True)

criterion = {
    'frame': getattr(nn, args.frame_loss['name'])(**args.frame_loss['args']).to(device),
    'utter': getattr(nn, args.utter_loss['name'])(**args.utter_loss['args']).to(device)
}


if args.train:
    dataset = {
        'train': HalfTruthDataset(**args.train_dataset_args),
        'valid': HalfTruthDataset(**args.valid_dataset_args)
    }

    loader = {
        'train': DataLoader(dataset['train'], batch_size=args.batch, num_workers=4, pin_memory=True, \
                collate_fn=dataset['train'].pad_batch, shuffle=True, drop_last=True),
        'valid': DataLoader(dataset['valid'], batch_size=args.batch, num_workers=4, pin_memory=True, \
                collate_fn=dataset['valid'].pad_batch)
    }
    
    optimizer = getattr(optim, args.optim['name'])(model.parameters(), **args.optim['args'])

    runner = Runner(model, loader, device, criterion, optimizer, args=args)
    runner.train()


if args.test:
    dataset = {
        'test': HalfTruthDataset(**args.test_dataset_args)
    }

    loader = {
        'test': DataLoader(dataset['test'], batch_size=args.batch, num_workers=4, pin_memory=True, \
                collate_fn=dataset['test'].pad_batch)
    }

    runner = Runner(model, loader, device, criterion, args=args)
    runner.test(args.params)