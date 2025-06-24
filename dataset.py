import numpy as np

import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset



class HalfTruthDataset(Dataset):
    def __init__(self, data_path, data_list, frame_size=400, frame_rate=320):
        super(HalfTruthDataset, self).__init__()

        with open(data_list, 'r') as in_file:
            infos = in_file.read().splitlines()

        info_dict = {}
        for info in infos:
            name, segments, label = info.split(' ')

            label = np.array(label, dtype=np.int64)

            if label == 1:
                stamp = np.array([[0, 0]], dtype=np.int64)
            else:
                stamp = []
                
                segments = segments.split('/')
                for segment in segments:
                    
                    if segment.endswith('T'):
                        continue
                
                    bgn, end, _ = segment.split('-')
                    stamp.append([bgn, end])
                
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
        
        wave, sr = torchaudio.load(path)
        if sr != 16000:
            wave = torchaudio.functional.resample(wave, sr, 16000)
        wave = wave[0]

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

        frame_y = torch.ones(batch_x.size(0), max(lengths), dtype=torch.int64)
        for idx, (stamp, label) in enumerate(zip(stamps, labels)):
            if label == 0:
                stamp = torch.ceil((stamp - self.frame_size).clamp(min=0) / self.frame_rate)
                stamp = stamp.type(torch.int64)
            
                for bgn, end in stamp:
                    frame_y[idx][bgn:end+1] = 0

        utter_y = torch.stack(labels)

        return names, batch_x, frame_y, utter_y, lengths


class CodecFakeDataset(Dataset):
    def __init__(self, data_path, data_list, codec_list, frame_size=400, frame_rate=320):
        super(CodecFakeDataset, self).__init__()

        with open(data_list, 'r') as in_file:
            self.names = in_file.read().splitlines()
        
        with open(codec_list, 'r') as in_file:
            self.codecs = in_file.read().splitlines()

        self.data_path  = data_path
        self.frame_size = frame_size
        self.frame_rate = frame_rate

    def __getitem__(self, idx):
        name = self.names[idx]
        
        paste_path = f'{self.data_path}/vctk_48k/{name}'
        paste_wave, sr = torchaudio.load(paste_path)
        
        if sr != 16000:
            paste_wave = torchaudio.functional.resample(paste_wave, sr, 16000)
        paste_wave = paste_wave[0]

        label = np.random.randint(2)
        if label == 1:
            stamp = np.array([[0, 0]], dtype=np.int64)
            
        else:
            stamp = []
            
            codec = np.random.choice(self.codecs)
            codec_path = f'{self.data_path}/{codec}/{name}'
            codec_wave, sr = torchaudio.load(codec_path)

            if sr != 16000:
                codec_wave = torchaudio.functional.resample(codec_wave, sr, 16000)
            codec_wave = codec_wave[0]

            wave_len = min(len(paste_wave), len(codec_wave))
            bgn = np.random.randint(wave_len)
            end = min(bgn + np.random.randint(1600, 160000), wave_len - 1) # 0.1~10s

            paste_wave[bgn:end] = codec_wave[bgn:end]
            stamp.append([bgn, end])

            if np.random.uniform(0, 1) > 0.9:
                # bgn = end
                bgn = np.randon.randint(end + 1, wave_len)
                end = min(bgn + np.random.randint(1600, 160000), wave_len - 1) # 0.1~10s
                
                paste_wave[bgn:end] = codec_wave[bgn:end]
                stamp.append([bgn, end])
                
            stamp = np.array(stamp, dtype=np.int64)

        stamp = torch.tensor(stamp, dtype=torch.int64)
        label = torch.tensor(label, dtype=torch.int64)
        
        return name, paste_wave, stamp, label
        
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

        frame_y = torch.ones(batch_x.size(0), max(lengths), dtype=torch.int64)
        for idx, (stamp, label) in enumerate(zip(stamps, labels)):
            if label == 0:
                stamp = torch.ceil((stamp - self.frame_size).clamp(min=0) / self.frame_rate)
                stamp = stamp.type(torch.int64)
            
                for bgn, end in stamp:
                    frame_y[idx][bgn:end+1] = 0

        utter_y = torch.stack(labels)

        return names, batch_x, frame_y, utter_y, lengths


class FixedCodecFakeDataset(Dataset):
    def __init__(self, data_path, data_list, codec, frame_size=400, frame_rate=320):
        super(FixedCodecFakeDataset, self).__init__()

        with open(data_list, 'r') as in_file:
            infos = in_file.read().splitlines()

        info_dict = {}
        for info in infos:
            name, segments, label = info.split(' ')

            label = np.array(label, dtype=np.int64)

            if label == 1:
                stamp = np.array([[0, 0]], dtype=np.int64)
            else:
                stamp = []
                
                segments = segments.split('/')
                for segment in segments:
                    
                    if segment.endswith('T'):
                        continue
                
                    bgn, end, _ = segment.split('-')
                    stamp.append([bgn, end])
                
                stamp = np.array(stamp, dtype=np.float32) * 16000
                stamp = stamp.astype(np.int64)

            path  = f'{data_path}/{name}.wav'
            stamp = torch.tensor(stamp, dtype=torch.int64)
            label = torch.tensor(label, dtype=torch.int64)

            info_dict[name] = {'path': path, 'stamp': stamp, 'label': label}

        self.names = list(info_dict.keys())

        self.codec      = codec
        self.data_path  = data_path
        self.info_dict  = info_dict
        self.frame_size = frame_size
        self.frame_rate = frame_rate

    def __getitem__(self, idx):
        name = self.names[idx]
        paste_path = f'{self.data_path}/vctk_48k/{name}'
        paste_wave, sr = torchaudio.load(paste_path)
        
        if sr != 16000:
            paste_wave = torchaudio.functional.resample(paste_wave, sr, 16000)
        paste_wave = paste_wave[0]

        stamp = self.info_dict[name]['stamp']
        label = self.info_dict[name]['label']

        if label == 0:
            codec_path = f'{self.data_path}/{self.codec}/{name}'
            codec_wave, sr = torchaudio.load(codec_path)
            
            if sr != 16000:
                codec_wave = torchaudio.functional.resample(codec_wave, sr, 16000)
            codec_wave = codec_wave[0]

            stamp[1] = min(stamp[1], len(paste_wave) - 1, len(codec_wave) - 1)
            bgn, end = stamp
            
            paste_wave[bgn:end+1] = codec_wave[bgn:end+1]

        return name, paste_wave, stamp, label

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
        
        frame_y = torch.ones(batch_x.size(0), max(lengths), dtype=torch.int64)
        for idx, (stamp, label) in enumerate(zip(stamps, labels)):
            if label == 0:
                stamp = torch.ceil((stamp - self.frame_size).clamp(min=0) / self.frame_rate)
                stamp = stamp.type(torch.int64)
            
                for bgn, end in stamp:
                    frame_y[idx][bgn:end+1] = 0

        utter_y = torch.stack(labels)

        return names, batch_x, frame_y, utter_y, lengths


class SINECodecFakeDataset(Dataset):
    def __init__(self, data_path, data_list, frame_size=400, frame_rate=320):
        super(SINECodecFakeDataset, self).__init__()

        with open(data_list, 'r') as in_file:
            infos = in_file.read().splitlines()
        self.names = [ info.split(' ')[0].split('-', 1)[1] for info in infos ]
        
        self.data_path  = data_path
        self.frame_size = frame_size
        self.frame_rate = frame_rate

    def __getitem__(self, idx):
        name = self.names[idx]
        
        paste_path = f'{self.data_path}/dev_real_medium-{name}.wav'
        paste_wave, sr = torchaudio.load(paste_path)
        
        if sr != 16000:
            paste_wave = torchaudio.functional.resample(paste_wave, sr, 16000)
        paste_wave = paste_wave[0]

        label = np.random.randint(2)
        if label == 1:
            stamp = np.array([[0, 0]], dtype=np.int64)

        else:
            stamp = []
            
            resyn_path = f'{self.data_path}/dev_resyn_medium-{name}.wav'
            resyn_wave, sr = torchaudio.load(resyn_path)

            if sr != 16000:
                resyn_wave = torchaudio.functional.resample(resyn_wave, sr, 16000)
            resyn_wave = resyn_wave[0]

            wave_len = min(len(paste_wave), len(resyn_wave))
            bgn = np.random.randint(wave_len)
            end = min(bgn + np.random.randint(800, 24000), wave_len - 1) # 0.05~1.5s

            paste_wave[bgn:end] = resyn_wave[bgn:end]
            stamp.append([bgn, end])

            # if np.random.uniform(0, 1) > 0.9:
            #     # bgn = end
            #     bgn = np.randon.randint(end + 1, wave_len)
            #     end = min(bgn + np.random.randint(800, 24000), wave_len - 1) # 0.05~1.5s
            #     
            #     paste_wave[bgn:end] = resyn_wave[bgn:end]
            #     stamp.append([bgn, end])
                
            stamp = np.array(stamp, dtype=np.int64)


        stamp = torch.tensor(stamp, dtype=torch.int64)
        label = torch.tensor(label, dtype=torch.int64)
        
        return name, paste_wave, stamp, label

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

        frame_y = torch.ones(batch_x.size(0), max(lengths), dtype=torch.int64)
        for idx, (stamp, label) in enumerate(zip(stamps, labels)):
            if label == 0:
                stamp = torch.ceil((stamp - self.frame_size).clamp(min=0) / self.frame_rate)
                stamp = stamp.type(torch.int64)
            
                for bgn, end in stamp:
                    frame_y[idx][bgn:end+1] = 0

        utter_y = torch.stack(labels)

        return names, batch_x, frame_y, utter_y, lengths



class PseudoEditDataset(Dataset):
    def __init__(self, data_path, data_list, frame_size=400, frame_rate=320):
        super(PseudoEditDataset, self).__init__()

        with open(data_list, 'r') as in_file:
            infos = in_file.read().splitlines()

        info_dict = {}
        for info in infos:
            name, segments, label = info.split(' ')

            label = np.array(label, dtype=np.int64)

            if label == 0 or label == 1:
                stamp = np.array([[0, 0]], dtype=np.int64)
            else:
                stamp = []
                
                segments = segments.split('/')
                for segment in segments:
                    
                    if segment.endswith('T'):
                        continue
                
                    bgn, end, _ = segment.split('-')
                    stamp.append([bgn, end])
                
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

        wave, sr = torchaudio.load(path)
        if sr != 16000:
            wave = torchaudio.functional.resample(wave, sr, 16000)
        wave = wave[0]

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
        
        frame_y = []
        utter_y = []
        for idx in range(len(labels)):
            if   labels[idx] == 0:
                frame_y_ = torch.zeros(max(lengths), dtype=torch.int64)
                utter_y_ = 0
            elif labels[idx] == 1:
                frame_y_ = torch.ones(max(lengths), dtype=torch.int64)
                utter_y_ = 1
            elif labels[idx] == 2:
                frame_y_ = torch.zeros(max(lengths), dtype=torch.int64)
                
                stamp = torch.ceil((stamps[idx] - self.frame_size).clamp(min=0) / self.frame_rate)
                stamp = stamp.type(torch.int64)
            
                for bgn, end in stamp:
                    frame_y_[bgn:end+1] = 2
                utter_y_ = 2
            else:
                frame_y_ = torch.ones(max(lengths), dtype=torch.int64)
                
                stamp = torch.ceil((stamps[idx] - self.frame_size).clamp(min=0) / self.frame_rate)
                stamp = stamp.type(torch.int64)
            
                for bgn, end in stamp:
                    frame_y_[bgn:end+1] = 2
                utter_y_ = 2

            frame_y.append(frame_y_)
            utter_y.append(utter_y_)
        frame_y = torch.stack(frame_y, dim=0)
        utter_y = torch.tensor(utter_y, dtype=torch.int64)

        return names, batch_x, frame_y, utter_y, lengths
