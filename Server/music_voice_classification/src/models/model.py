import torch
import torch.nn as nn
import torch.nn.functional as F

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
print("cpu와 cuda 중 다음 기기로 학습함:", DEVICE)

class MusicGenre(nn.Module):
    def training_step(self, batch):
        audios, labels = batch
        audios = audios.to(DEVICE)
        labels = labels.to(DEVICE)
        out = self(audios)
        loss = F.cross_entropy(out, labels)
        return loss

    def validation_step(self, batch):
        audios, labels = batch
        audios = audios.to(DEVICE)
        labels = labels.to(DEVICE)
        out = self(audios)
        loss = F.cross_entropy(out, labels)
        _, preds = torch.max(out, dim=1)
        acc = torch.tensor(torch.sum(preds == labels).item() / len(preds))
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_acc']))



def conv_block_1(in_channels, out_channels, kernel_size, padding, pool=False):
    layers = [nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
              nn.BatchNorm1d(out_channels),
              nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool1d(100))
    return nn.Sequential(*layers)

def conv_block_2(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class MusicCNN(MusicGenre):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.conv1 = conv_block_1(in_channels, 32, 11, 5, pool=True)
        self.conv2 = conv_block_1(in_channels, 32, 19, 10, pool=True)
        self.conv3 = conv_block_1(in_channels, 32, 39, 20, pool=True)

        self.conv4 = conv_block_2(in_channels=3, out_channels=64)
        self.conv5 = conv_block_2(in_channels=64, out_channels=128, pool=True)
        self.conv6 = conv_block_2(in_channels=128, out_channels=128)
        self.conv7 = conv_block_2(in_channels=128, out_channels=256, pool=True)
        self.conv8 = conv_block_2(in_channels=256, out_channels=256, pool=True)
        self.classifier = nn.Sequential(nn.MaxPool2d(kernel_size=(4, 4), stride=(4, 4)),
                                        nn.Flatten(),
                                        nn.Dropout(0.25))

        self.fc1 = nn.Linear(3840, 1000)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(1000, 100)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(100, num_classes)
        self.logSoftmax = nn.LogSoftmax(dim=1)


    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x_new = torch.stack((x1, x2, x3), dim=1)

        x = self.conv4(x_new)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.classifier(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.logSoftmax(x)
        return x
