import os
import torch
import torchaudio
from torchaudio import transforms
from torch.utils.data import random_split
from torch.utils.data import Dataset, DataLoader

new_dir = '../content/audio_3_sec'
val_size = 5   # validation 개수 지정 보통 train : val = 4 : 1 비율
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


    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        fname = self.files[i]                   # fname blues_10.wav
        fpath = os.path.join(self.root, fname)  # fpath ./content/audio3sec/blues\blues_10.wav

        '''
            data, sample_rate = torchaudio.load('foo.mp3')
            >>> print(data.size())
            torch.Size([2, 278756])
            >>> print(sample_rate)
            44100
        '''

        audio, sr = torchaudio.load(fpath, normalize=True)                  # audio tensor([[ 0.0073,  0.0166,  0.0076,  ..., -0.0437, -0.0571, -0.0409]])
        # print(audio.shape, sr)                      # torch.Size([2, 132300]) 44100: [채널 수, 데이터 길이], sampling rate
        # print(audio.shape[1] / sr)                  # 음성 파일 시간: 3초

        # if sr != 48000:                                 # sampling rate를 48000으로 통일
        #     # print(audio.shape)
        #     transform = transforms.Resample(sr, 48000)
        #     audio = transform(audio)                     # (1, 144000)
        #
        # if audio.shape[0] != 2:
        #     zero = torch.zeros_like(audio)               # (1, 144000)
        #     audio = torch.stack([audio, zero])           # (2, 1, 144000)
        #     audio = audio.squeeze(1)                     # (2, 144000)

        # print(audio.view([-1, 2]).shape)
        # audio = audio.view([-1, 2])             # (72000, 2)
        # audio = torch.transpose(audio, 0, 1)    # (2, 72000)
        # audio = audio.view([-1, 2])
        # audio = torch.transpose(audio, 0, 1)
        # print('change', audio.shape)
        # if audio.shape[1] == 132300:                         # channel을 2로 통일
        #     # print(audio.view([-1, 2]).shape)            # torch.Size([66150, 2])
        #     audio = audio.view([-1, 2]).shape
        # elif audio.shape[1] == 14400:
        #     # print(audio.view([-1, 2]).shape)            # torch.Size([72000, 2])
        #     audio = audio.view([-1, 2]).shape

        if sr != 48000:
            transform = transforms.Resample(sr, 48000)
            audio = transform(audio)
        class_idx = self.classes.index(parse_genres(fname))                 # class_idx 0 or 1
        return audio, class_idx



dataset = MusicDataset(new_dir)
print('dataset length: ', len(dataset))
print('dataset[0]: ', dataset[0])
for data in dataset:
    print(data[0].shape)

random_seed = 42
torch.manual_seed(random_seed)
train_size = len(dataset) - val_size

train_ds, val_ds = random_split(dataset, [train_size, val_size])
print('train, val length:', len(train_ds), len(val_ds))


train_dl = DataLoader(train_ds, batch_size, shuffle=True)
valid_dl = DataLoader(val_ds, batch_size*2)
test_dl = DataLoader(val_ds, 1, shuffle=True)
