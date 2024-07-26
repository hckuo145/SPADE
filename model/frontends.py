import einops
import s3prl.hub as hub

import torch
import torch.nn as nn
import torchaudio


class MSTFT(nn.Module):
    def __init__(self, frame_size=400, frame_rate=320, num_mels=80):
        super(MSTFT, self).__init__()

        self.extractor = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft     =frame_size,
            hop_length=frame_rate,
            n_mels    =num_mels,
            window_fn =torch.hamming_window,
            center    =False
        )
    
    def forward(self, waves):
        with torch.no_grad():
            feats = self.extractor(waves)
            feats = einops.rearrange(feats, 'b f t -> b t f')
            feats = nn.functional.layer_norm(feats, (feats.size(-1),))

        return feats


class Wav2Vec2_randinit(nn.Module):
    def __init__(self, weighted_sum=True, requires_grad=False):
        super(Wav2Vec2_randinit, self).__init__()

        self.extractor = torchaudio.models.wav2vec2_base()
        self.extractor.requires_grad_(requires_grad)

        if weighted_sum:
            self.weights = nn.Parameter(torch.ones(12) / 12)
        self.weighted_sum = weighted_sum

    def forward(self, waves):
        feats = self.extractor.extract_features(waves)[0]
        
        if self.weighted_sum:
            feats = torch.stack(feats, dim=0)
            feats = nn.functional.layer_norm(feats, (feats.size(-1),))
            feats = torch.einsum('n, n b t h -> b t h', self.weights.softmax(dim=-1), feats)

        else:
            feats = feats[-1]
            feats = nn.functional.layer_norm(feats, (feats.size(-1),))

        return feats


class Wav2Vec2(nn.Module):
    def __init__(self, weighted_sum=True, requires_grad=False):
        super(Wav2Vec2, self).__init__()

        self.extractor = hub.wav2vec2()
        self.extractor.requires_grad_(requires_grad)

        if weighted_sum:
            self.weights = nn.Parameter(torch.ones(13) / 13)
        self.weighted_sum = weighted_sum

    def forward(self, waves):
        if self.weighted_sum:
            feats = self.extractor(waves)['hidden_states']
            feats = torch.stack(feats, dim=0)

            feats = nn.functional.layer_norm(feats, (feats.size(-1),))
            feats = torch.einsum('n, n b t h -> b t h', self.weights.softmax(dim=-1), feats)

        else:
            feats = self.extractor(waves)['last_hidden_state']
            feats = nn.functional.layer_norm(feats, (feats.size(-1),))


        return feats
    

class HuBERT(nn.Module):
    def __init__(self, weighted_sum=True, requires_grad=False):
        super(HuBERT, self).__init__()

        self.extractor = hub.hubert()
        self.extractor.requires_grad_(requires_grad)

        if weighted_sum:
            self.weights = nn.Parameter(torch.ones(13) / 13)
        self.weighted_sum = weighted_sum

    def forward(self, waves):
        if self.weighted_sum:
            feats = self.extractor(waves)['hidden_states']
            feats = torch.stack(feats, dim=0)

            feats = nn.functional.layer_norm(feats, (feats.size(-1),))
            feats = torch.einsum('n, n b t h -> b t h', self.weights.softmax(dim=-1), feats)

        else:
            feats = self.extractor(waves)['last_hidden_state']
            feats = nn.functional.layer_norm(feats, (feats.size(-1),))

        return feats
    

class WavLM(nn.Module):
    def __init__(self, weighted_sum=True, requires_grad=False):
        super(WavLM, self).__init__()

        self.extractor = hub.wavlm()
        self.extractor.requires_grad_(requires_grad)

        if weighted_sum:
            self.weights = nn.Parameter(torch.ones(13) / 13)
        self.weighted_sum = weighted_sum

    def forward(self, waves):
        if self.weighted_sum:
            feats = self.extractor(waves)['hidden_states']
            feats = torch.stack(feats, dim=0)

            feats = nn.functional.layer_norm(feats, (feats.size(-1),))
            feats = torch.einsum('n, n b t h -> b t h', self.weights.softmax(dim=-1), feats)

        else:
            feats = self.extractor(waves)['last_hidden_state']
            feats = nn.functional.layer_norm(feats, (feats.size(-1),))

        return feats
    

if __name__ == '__main__':
    import sys
    sys.path.append('/work/hckuo145/SPADE')

    from dataset import HalfTruthDataset
    from torch.utils.data import DataLoader

    # frontend = MSTFT()
    frontend = Wav2Vec2(weighted_sum=True, requires_grad=False)
    # frontend = HuBERT(weighted_sum=True, requires_grad=False)
    # frontend = WavLM(weighted_sum=True, requires_grad=False)
    # frontend = Wav2Vec2_randinit(weighted_sum=True, requires_grad=False)
    print(frontend.extractor.requires_grad)
    print(frontend.weights.requires_grad)
    # split = 'train'
    # data_path = f'/work/hckuo145/SPADE/data/HAD/HAD_{split}/{split}'
    # data_list = f'/work/hckuo145/SPADE/data/HAD/HAD_{split}/HAD_{split}_label.txt'

    # dataset = HalfTruthDataset(data_path, data_list)
    # loader  = DataLoader(dataset, batch_size=10, collate_fn=dataset.pad_batch)

    # for i, (names, waves, stamps, labels, nframes) in enumerate(loader):
    #     print(waves.size())
    #     feats = frontend(waves)
    #     print(feats.size())
    #     break