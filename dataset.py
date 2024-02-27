import numpy as np

import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset


class HalfTruthDataset(Dataset):
    def __init__(self, data_path, data_list, frame_size=400, frame_rate=320):
        super(HalfTruthDataset, self).__init__()

        with open(data_list, 'r') as in_file:
            infos = in_file.readlines()

        info_dict = {}
        for info in infos:
            name, segments, label = info.split(' ')

            stamp = [0, 0]
            stamp = np.array(stamp, dtype=np.int64)
            label = np.array(label, dtype=np.int64)

            segments = segments.split('/')
            for segment in segments:
                
                if segment.endswith('T'):
                    continue
            
                bgn, end, _ = segment.split('-')

                stamp = [bgn, end]
                stamp = np.array(stamp, dtype=np.float32) * 16000
                stamp = stamp.astype(np.int64)

            path  = f'{data_path}/{name}.wav'
            stamp = torch.tensor(stamp, dtype=torch.int64)
            label = torch.tensor(label, dtype=torch.int64)

            info_dict[name] = {'path': path, 'stamp': stamp, 'label': label}

        self.names = list(info_dict.keys())

        self.info_dict  = info_dict
        self.frame_size = frame_size
        self.frame_rate = frame_rate

    def __getitem__(self, idx):
        name = self.names[idx]
        path = self.info_dict[name]['path']
        wave = torchaudio.load(path)[0][0]

        stamp = self.info_dict[name]['stamp']
        label = self.info_dict[name]['label']

        return name, wave, stamp, label

    def __len__(self):
        return len(self.names)
    
    def pad_batch(self, batch):
        names, waves, stamps, labels = zip(*batch)
        
        lengths = torch.tensor([ wave.size(0) for wave in waves ], dtype=torch.float32)
        lengths = torch.ceil((lengths - self.frame_size).clamp(min=0) / self.frame_rate) + 1
        lengths = lengths.type(torch.int64)
        padding = (max(lengths) - 1) * self.frame_rate + self.frame_size

        batch_x = [ nn.functional.pad(wave, (0, padding - wave.size(0))) for wave in waves ]
        batch_x = torch.stack(batch_x, dim=0)
        
        stamps = torch.stack(stamps, dim=0)
        stamps = torch.ceil((stamps - self.frame_size).clamp(min=0) / self.frame_rate)
        stamps = stamps.type(torch.int64)

        frame_y = torch.ones(batch_x.size(0), max(lengths), dtype=torch.int64)
        for idx, label in enumerate(labels):
            bgn, end = stamps[idx]
            
            if label == 0:
                frame_y[idx][bgn:end+1] = 0

        utter_y = torch.stack(labels)

        return names, batch_x, frame_y, utter_y, lengths


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    split = 'train'
    data_path = f'/home/hckuo/project/spade/data/HAD/HAD_{split}/{split}'
    data_list = f'/home/hckuo/project/spade/data/HAD/HAD_{split}/HAD_{split}_label.txt'

    dataset = HalfTruthDataset(data_path, data_list)
    loader  = DataLoader(dataset, batch_size=256, collate_fn=dataset.pad_batch)

    for names, waves, stamps, labels, nframes in loader:
        print(waves.size(), stamps.size(), labels.size(), nframes.size())