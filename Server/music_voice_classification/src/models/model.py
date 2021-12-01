import torch
import torch.nn as nn
import torch.nn.functional as F

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")
print("cpu와 cuda 중 다음 기기로 학습함:", DEVICE)

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class MusicGenreClassificationBase(nn.Module):
    def training_step(self, batch):
        audios, labels = batch
        audios = audios.to(DEVICE)
        labels = labels.to(DEVICE)
        out = self(audios)                   # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        return loss

    def validation_step(self, batch):
        audios, labels = batch
        audios = audios.to(DEVICE)
        labels = labels.to(DEVICE)
        out = self(audios)                   # Generate predictions
        loss = F.cross_entropy(out, labels)  # Calculate loss
        acc = accuracy(out, labels)          # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_acc']))


# ((W-K+2*P)/S)+1 66150-3+2 /
def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


def conv_block1(in_channels, out_channels, pool=False):
    layers = [nn.Conv1d(in_channels, out_channels, kernel_size=11, padding=5),
              nn.BatchNorm1d(out_channels),
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool1d(100))
    return nn.Sequential(*layers)


def conv_block2(in_channels, out_channels, pool=False):
    layers = [nn.Conv1d(in_channels, out_channels, kernel_size=51, padding=30),
              nn.BatchNorm1d(out_channels),
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool1d(100))
    return nn.Sequential(*layers)


def conv_block3(in_channels, out_channels, pool=False):
    layers = [nn.Conv1d(in_channels, out_channels, kernel_size=101, padding=50),
              nn.BatchNorm1d(out_channels),
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool1d(100))
    return nn.Sequential(*layers)



class multitemporalfeturemap(MusicGenreClassificationBase):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.conv1 = conv_block1(in_channels, 32, pool=True)
        self.conv2 = conv_block2(in_channels, 32, pool=True)
        self.conv3 = conv_block3(in_channels, 32, pool=True)

        self.conv_1 = conv_block(3, 64)
        self.conv_2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        self.conv_3 = conv_block(128, 256, pool=True)
        self.conv_4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        self.classifier = nn.Sequential(nn.MaxPool2d(4),
                                        nn.Flatten(),
                                        nn.Dropout(0.2))

        self.linear1 = nn.Linear(7680, 2000)
        # output layer
        self.linear2 = nn.Linear(2000, 100)
        self.linear3 = nn.Linear(100, num_classes)

    def forward(self, xb):
        out1 = self.conv1(xb)
        out2 = self.conv2(xb)
        out3 = self.conv3(xb)
        out_new = torch.stack((out1, out2, out3), dim=1)
        out = self.conv_1(out_new)
        out = self.conv_2(out)
        out = self.res1(out) + out
        out = self.conv_3(out)
        out = self.conv_4(out)
        out = self.res2(out) + out
        out_1 = self.classifier(out)
        xb = out_1.view(out_1.size(0), -1)
        out = self.linear1(xb)
        out = F.relu(out)
        out = self.linear2(out)
        out = F.relu(out)
        out = self.linear3(out)
        return out

    # def forward(self, xb):
    #     print("연산 전", xb.size())
    #     out1 = self.conv1(xb)
    #     print("conv1 연산 후", out1.size())
    #     out2 = self.conv2(xb)
    #     print("conv2 연산 후", out2.size())
    #     out3 = self.conv3(xb)
    #     print("conv3 연산 후", out3.size())
    #     out_new = torch.stack((out1, out2, out3), dim=1)
    #     print("out_new 연산 후", out_new.size())
    #     out = self.conv_1(out_new)
    #     print("conv_1 연산 후", out.size())
    #     out = self.conv_2(out)
    #     print("conv_2 연산 후", out.size())
    #     out = self.res1(out) + out
    #     print("res1 + out 연산 후", out.size())
    #     out = self.conv_3(out)
    #     print("conv_3 연산 후", out.size())
    #     out = self.conv_4(out)
    #     print("conv_4 연산 후", out.size())
    #     out = self.res2(out) + out
    #     print("res2 + out 연산 후", out.size())
    #     out_1 = self.classifier(out)
    #     print("classifier 연산 후", out_1.size())
    #     xb = out_1.view(out_1.size(0), -1)
    #     print("view 연산 후", xb.size())
    #     out = self.linear1(xb)
    #     print("linear 연산 후", out.size())
    #     out = F.relu(out)
    #     out = self.linear2(out)
    #     print("linear2 연산 후", out.size())
    #     out = F.relu(out)
    #     out = self.linear3(out)
    #     print("linear3 연산 후", out.size())
    #     return out