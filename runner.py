import os
import torch
import einops
import numpy as np
from tqdm            import tqdm
from collections     import defaultdict
from tensorboardX    import SummaryWriter
from sklearn.metrics import confusion_matrix


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

            if (self.monitor[mtrc]['mode'] == 'min' and self.metrics[mtrc] <= self.monitor[mtrc]['record']) or \
               (self.monitor[mtrc]['mode'] == 'max' and self.metrics[mtrc] >= self.monitor[mtrc]['record']):

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

    def _forward_step(self, batch_x, frame_y, utter_y, mask, phase='train', drop_last=True):
        frame_p, utter_p = self.model(batch_x, mask)
        
        frame_p = einops.rearrange(frame_p, 'b t d -> (b t) d')
        frame_y = einops.rearrange(frame_y, 'b t -> (b t)')
        mask    = einops.rearrange(~mask  , 'b t -> (b t)')

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

        frame_p = einops.rearrange(frame_p, '(b t) d -> b t d', b=batch_x.size(0))

        return frame_p, utter_p

    def train(self):
        while self.epoch <= self.max_epoch:
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
                    mask = torch.arange(max(lengths))[None, :] >= lengths[:, None]
                    
                    mask    = mask.to(self.device)
                    batch_x = batch_x.to(self.device)
                    frame_y = frame_y.to(self.device)
                    utter_y = utter_y.to(self.device)
                    
                    if phase == 'train':
                        frame_p, utter_p = self._forward_step(batch_x, frame_y, utter_y, mask, phase)
                    else:
                        with torch.no_grad():
                            frame_p, utter_p = self._forward_step(batch_x, frame_y, utter_y, mask, phase, drop_last=False)

                    for true, pred, l in zip(frame_y, frame_p, lengths):
                        true, pred = true[:l], torch.argmax(pred[:l], dim=-1)
                        frame_true += list(true.detach().cpu().numpy())
                        frame_pred += list(pred.detach().cpu().numpy())

                    utter_p = torch.argmax(utter_p, dim=-1)
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

    @torch.no_grad()
    def test(self, checkpoint):
        self.load_checkpoint(checkpoint, params_only=True)

        frame_true, frame_pred = [], []
        utter_true, utter_pred = [], [] 

        self.model.eval()
        for _, batch_x, frame_y, utter_y, lengths in tqdm(self.loader['test'], ncols=100):
            mask = torch.arange(max(lengths))[None, :] >= lengths[:, None]

            mask    = mask.to(self.device)
            batch_x = batch_x.to(self.device)
            frame_y = frame_y.to(self.device)
            utter_y = utter_y.to(self.device)

            frame_p, utter_p = self._forward_step(batch_x, frame_y, utter_y, mask, phase='test')
 
            for true, pred, l in zip(frame_y, frame_p, lengths):
                true, pred = true[:l], torch.argmax(pred[:l], dim=-1)
                frame_true += list(true.detach().cpu().numpy())
                frame_pred += list(pred.detach().cpu().numpy())

            utter_p = torch.argmax(utter_p, dim=-1)
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