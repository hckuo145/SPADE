import os
import yaml
import einops
import argparse
import numpy as np
from tqdm            import tqdm
from collections     import defaultdict
from tensorboardX    import SummaryWriter
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model   import ConvGRU
from dataset import HalfTruthDataset


class Runner():
    def __init__(self, model, loader, device, criterion=None, optimizer=None, args=None):
        self.model     = model
        self.loader    = loader
        self.device    = device
        self.criterion = criterion
        self.optimizer = optimizer

        self.epoch = 0
        self.metrics = defaultdict(float)

        if args.train:
            self.writer = SummaryWriter(f'{args.exp_path}/{args.title}/tensorboard')

        if args.max_epoch == -1 and args.patience != -1:
            args.max_epoch = np.iinfo(int).max

        for mtrc in args.monitor.keys():
            if args.monitor[mtrc]['mode'] == 'min':
                args.monitor[mtrc]['record'] = np.finfo(float).max
            else:
                args.monitor[mtrc]['record'] = np.finfo(float).min

            args.monitor[mtrc]['count'] = 0

        vars(self).update({ key: value for key, value in vars(args).items() if key not in dir(self) })

    def save_checkpoint(self, checkpoint):
        state_dict = {'epoch': self.epoch}

        attrs = ['model', 'criterion', 'optimizer']
        for attr in attrs:
            if isinstance(getattr(self, attr), dict):
                for name in getattr(self, attr).keys():
                    state_dict[f'{attr}_{name}'] = getattr(self, attr)[name].state_dict()
            else:
                state_dict[attr] = getattr(self, attr).state_dict()

        os.makedirs(os.path.dirname(checkpoint), exist_ok=True)
        torch.save(state_dict, checkpoint)

    def load_checkpoint(self, checkpoint, params_only=False):
        state_dict = torch.load(checkpoint, map_location=self.device)

        if params_only:
            attrs = ['model']
        else:
            self.epoch = state_dict['epoch']
            attrs = ['model', 'criterion', 'optimizer']

        for attr in attrs:
            if isinstance(getattr(self, attr), dict):
                for name in getattr(self, attr).keys():
                    getattr(self, attr)[name].load_state_dict(state_dict[f'{attr}_{name}'])
            else:
                getattr(self, attr).load_state_dict(state_dict[f'{attr}'])

    def _update_callback(self, save_best_only=True):
        if not save_best_only:
            self.save_checkpoint(f'{self.exp_path}/{self.title}/checkpoints/epoch_{self.epoch}.pt')

        for mtrc in self.monitor.keys():
            assert mtrc in self.metrics.keys()

            if (self.monitor[mtrc]['mode'] == 'min' and self.metrics[mtrc] < self.monitor[mtrc]['record']) or \
               (self.monitor[mtrc]['mode'] == 'max' and self.metrics[mtrc] > self.monitor[mtrc]['record']):

                self.monitor[mtrc]['record'] = self.metrics[mtrc]
                self.monitor[mtrc]['count']  = 0
                
                self.save_checkpoint(f'{self.exp_path}/{self.title}/checkpoints/best_{mtrc.replace("/", "_")}.pt')

            else:
                self.monitor[mtrc]['count'] += 1

    def _check_early_stopping(self):
        return self.patience != -1 and self.epoch >= self.min_epoch and \
               all([ info['count'] >= self.patience for info in self.monitor.values() ])

    def _write_to_tensorboard(self, iteration):
        for key, value in self.metrics.items():
            self.writer.add_scalar(key, value, iteration)

    def _display(self, phase='train', iteration=None):
        disp = f'[{phase}]'

        if iteration is not None:
            disp += f' Iter {iteration}'

        for key, value in self.metrics.items():
            if isinstance(value, float):
                if key.endswith('loss'):
                    disp += f' - {key.replace("/", "_")}: {value:4.3e}'
                else:
                    disp += f' - {key.replace("/", "_")}: {value:5.4f}'
            else:
                    disp += f' - {key.replace("/", "_")}: {value}'

        print(disp, flush=True)

    def _forward_step(self, batch_x, frame_y, utter_y, lengths, phase='train', drop_last=True):
        mask = (torch.arange(max(lengths))[None, :] < lengths[:, None]).to(batch_x.device)
        
        frame_p = self.model(batch_x, lengths)
        frame_p = torch.softmax(frame_p, dim=2)[..., 0]
        frame_p = frame_p.masked_fill(~mask, 0)
        
        utter_p = torch.sum(frame_p ** 2, dim=1) / torch.sum(frame_p, dim=1)

        mask    = einops.rearrange(mask   , 'b t -> (b t)')
        frame_p = einops.rearrange(frame_p, 'b t -> (b t)')
        frame_y = einops.rearrange(frame_y, 'b t -> (b t)')

        frame_y, utter_y = 1. - frame_y, 1. - utter_y
        frame_loss = self.criterion['frame'](frame_p[mask], frame_y[mask])
        utter_loss = self.criterion['utter'](utter_p, utter_y)
        
        loss = frame_loss + utter_loss

        if phase == 'train':
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        reduce = len(self.loader[phase]) if drop_last else \
                len(self.loader[phase].dataset) / batch_x.size(0)
        self.metrics[f'{phase}/loss']       += loss.item()       / reduce
        self.metrics[f'{phase}/frame_loss'] += frame_loss.item() / reduce
        self.metrics[f'{phase}/utter_loss'] += utter_loss.item() / reduce

        frame_p = einops.rearrange(frame_p, '(b t) -> b t', b=batch_x.size(0))

        return frame_p, utter_p

    def train(self):
        while self.epoch < self.max_epoch:
            self.epoch  += 1
            self.metrics = defaultdict(float)

            for phase in ['train', 'valid']:
                frame_true, frame_pred = [], []
                utter_true, utter_pred = [], [] 

                if phase == 'train':
                    self.model.train()
                else:
                    self.model.eval()

                for _, batch_x, frame_y, utter_y, lengths in tqdm(self.loader[phase], ncols=100):
                    batch_x = batch_x.to(self.device)
                    frame_y = frame_y.to(self.device)
                    utter_y = utter_y.to(self.device)
                    
                    if phase == 'train':
                        frame_p, utter_p = self._forward_step(batch_x, frame_y, utter_y, lengths, phase)
                    else:
                        with torch.no_grad():
                            frame_p, utter_p = self._forward_step(batch_x, frame_y, utter_y, lengths, phase, drop_last=False)

                    for true, pred, l in zip(frame_y, frame_p, lengths):
                        true, pred = true[:l], (pred[:l] < 0.5).long()
                        frame_true += list(true.detach().cpu().numpy())
                        frame_pred += list(pred.detach().cpu().numpy())

                    utter_p = (utter_p < 0.5).long()
                    utter_true += list(utter_y.detach().cpu().numpy())
                    utter_pred += list(utter_p.detach().cpu().numpy())

                self.metrics[f'{phase}/frame_TP'], self.metrics[f'{phase}/frame_FN'], self.metrics[f'{phase}/frame_FP'], self.metrics[f'{phase}/frame_TN'], \
                self.metrics[f'{phase}/frame_A' ], self.metrics[f'{phase}/frame_P' ], self.metrics[f'{phase}/frame_R' ], self.metrics[f'{phase}/frame_F1'] = self.evaluate(frame_true, frame_pred)
                self.metrics[f'{phase}/utter_TP'], self.metrics[f'{phase}/utter_FN'], self.metrics[f'{phase}/utter_FP'], self.metrics[f'{phase}/utter_TN'], \
                self.metrics[f'{phase}/utter_A' ], self.metrics[f'{phase}/utter_P' ], self.metrics[f'{phase}/utter_R' ], self.metrics[f'{phase}/utter_F1'] = self.evaluate(utter_true, utter_pred)
                self.metrics[f'{phase}/score'] = 0.7 * self.metrics[f'{phase}/frame_F1'] + 0.3 * self.metrics[f'{phase}/utter_A']

            self._display('Train', self.epoch)
            self._write_to_tensorboard(self.epoch)
            self._update_callback(self.save_best_only)
            if self._check_early_stopping(): 
                break
        
        self.save_checkpoint(f'{self.exp_path}/{self.title}/checkpoints/last.pt')

    @torch.no_grad()
    def test(self, checkpoint):
        self.load_checkpoint(checkpoint, params_only=True)

        frame_true, frame_pred = [], []
        utter_true, utter_pred = [], [] 

        self.model.eval()
        for _, batch_x, frame_y, utter_y, lengths in tqdm(self.loader['test'], ncols=100):
            batch_x = batch_x.to(self.device)
            frame_y = frame_y.to(self.device)
            utter_y = utter_y.to(self.device)

            frame_p, utter_p = self._forward_step(batch_x, frame_y, utter_y, lengths, phase='test')
 
            for true, pred, l in zip(frame_y, frame_p, lengths):
                true, pred = true[:l], (pred[:l] < 0.5).long()
                frame_true += list(true.detach().cpu().numpy())
                frame_pred += list(pred.detach().cpu().numpy())

            utter_p = (utter_p < 0.5).long()
            utter_true += list(utter_y.detach().cpu().numpy())
            utter_pred += list(utter_p.detach().cpu().numpy())

        self.metrics['test/frame_TP'], self.metrics['test/frame_FN'], self.metrics['test/frame_FP'], self.metrics['test/frame_TN'], \
        self.metrics['test/frame_A' ], self.metrics['test/frame_P' ], self.metrics['test/frame_R' ], self.metrics['test/frame_F1'] = self.evaluate(frame_true, frame_pred)
        self.metrics['test/utter_TP'], self.metrics['test/utter_FN'], self.metrics['test/utter_FP'], self.metrics['test/utter_TN'], \
        self.metrics['test/utter_A' ], self.metrics['test/utter_P' ], self.metrics['test/utter_R' ], self.metrics['test/utter_F1'] = self.evaluate(utter_true, utter_pred)
        self.metrics['test/score'] = 0.7 * self.metrics['test/frame_F1'] + 0.3 * self.metrics['test/utter_A']

        self._display('Test')

    @staticmethod
    def evaluate(y_true, y_pred):
        # "fake" is labelled 0 but taken as positive
        TP, FN, FP, TN = confusion_matrix(y_true, y_pred).ravel()
        
        A  = (TP + TN) / (TP + TN + FP + FN)
        if TP == 0:
            P, R, F1 = 0, 0, 0
        else:
            P  = TP / (TP + FP)
            R  = TP / (TP + FN)
            F1 = 2 * P * R / (P + R)

        return TP, FN, FP, TN, A, P, R, F1
    

if __name__ == '__main__':
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