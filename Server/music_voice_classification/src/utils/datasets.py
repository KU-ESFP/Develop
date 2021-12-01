import os
import torch
import torchaudio
import librosa
import numpy as np
from torchaudio import transforms
from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader

# new_dir = '../content/test'
new_dir = '../content/train_datasets/audio_3_sec'
val_size = 200   # validation 개수 지정 보통 train : val = 4 : 1 비율
batch_size = 32

def parse_genres(fname):
    parts = fname.split('_')
    return ' '.join(parts[:-1])


class MusicDataset(Dataset):
    def __init__(self, root):
        super().__init__()
        self.root = root
        self.files = [fname for fname in os.listdir(root) if fname.endswith('.wav')]
        self.classes = list(set(parse_genres(fname) for fname in self.files))
        # self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        fname = self.files[i]                   # fname blues_10.wav
        # print(fname)
        fpath = os.path.join(self.root, fname)  # fpath ./content/audio3sec/blues\blues_10.wav
        # img = self.transform(open_image(fpath))
        '''
            data, sample_rate = torchaudio.load('foo.mp3')
            >>> print(data.size())
            torch.Size([2, 278756])
            >>> print(sample_rate)
            44100
        '''
        # audio = torchaudio.load(fpath)[0]                       # audio tensor([[ 0.0073,  0.0166,  0.0076,  ..., -0.0437, -0.0571, -0.0409]])
        audio, sr = torchaudio.load(fpath)
        # print(audio.shape, sr)                      # torch.Size([2, 132300]) 44100: [채널 수, 데이터 길이], sampling rate
        # print(audio.shape[1] / sr)                  # 음성 파일 시간: 3초

        class_idx = self.classes.index(parse_genres(fname))     # class_idx 0 or 1
        return audio, class_idx



dataset = MusicDataset(new_dir)
print('dataset length: ', len(dataset))

random_seed = 42
torch.manual_seed(random_seed)
train_size = len(dataset) - val_size

train_ds, val_ds = random_split(dataset, [train_size, val_size])
print('train, val length:', len(train_ds), len(val_ds))


train_dl = DataLoader(train_ds, batch_size, shuffle=True)
valid_dl = DataLoader(val_ds, batch_size)
test_dl = DataLoader(val_ds, 1, shuffle=True)
