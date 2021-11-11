import torch.nn as nn
import torch.nn.functional as F

'''
    [Convolution] - (1, 48, 48)
    W: 너비
    K: kernel_size
    P: padding
    S: stride
    = (W - K + P * 2) / S + 1
    
    [모델 레이어 쌓는 최적 순서]
    Convolution > Batch Normalization > Activation(relu, tanh) > Dropout > Pooling(max, avg)

'''

# Define a convolution neural network
class Network(nn.Module):
    def __init__(self, num_classes):
        super(Network, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=12, kernel_size=5)           # 12@44*44
        self.bn1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=5)          # 12@40*40
        self.bn2 = nn.BatchNorm2d(12)
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))                     # 24@20*20

        self.conv4 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5)          # 24@16*16
        self.bn4 = nn.BatchNorm2d(24)
        self.conv5 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=5)          # 24@12*12
        self.bn5 = nn.BatchNorm2d(24)

        self.fc1 = nn.Linear(24 * 12 * 12, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))

        x = x.view(-1, 24 * 12 * 12)
        x = self.fc1(x)
        return x


class LeNet(nn.Module):
    def __init__(self, num_classes):
        super(LeNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=20, kernel_size=(5, 5))      # 20@44*44
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))                 # 20@22*22

        self.conv2 = nn.Conv2d(in_channels=20, out_channels=50, kernel_size=(5, 5))     # 50@18*18
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))                 # 50@9*9

        self.fc1 = nn.Linear(50*9*9, 500)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(500, num_classes)
        self.logSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)

        x = x.view(-1, 50 * 9 * 9)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.logSoftmax(x)
        return x


class EmoModel(nn.Module):
    def __init__(self, num_classes):
        super(EmoModel, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3))      # 32@46*46
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3))     # 64@44*44
        self.relu2 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))                 # 64@22*22
        self.dropout1 = nn.Dropout(p=0.25)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3))    # 128@20*20
        self.relu3 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))                 # 128@10*10
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3))   # 128@8*8
        self.relu4 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))                 # 128@4*4
        self.dropout2 = nn.Dropout(p=0.25)

        self.fc1 = nn.Linear(128 * 4 * 4, 1024)
        self.relu5 = nn.ReLU()
        self.dropout3 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1024, num_classes)
        self.logSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool1(x)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool2(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.maxpool3(x)
        x = self.dropout2(x)

        x = x.view(-1, 128 * 4 * 4)
        x = self.fc1(x)
        x = self.relu5(x)
        x = self.dropout3(x)
        x = self.fc2(x)
        x = self.logSoftmax(x)
        return x

class CNN1(nn.Module):
    def __init__(self, num_classes):
        super(CNN1, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3))  # 32@46*46
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3))  # 64@44*44
        self.relu2 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))  # 64@22*22

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3))  # 128@20*20
        self.relu3 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))  # 128@10*10

        self.fc1 = nn.Linear(128 * 10 * 10, 1024)
        self.relu4 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.5)

        self.fc2 = nn.Linear(1024, 512)
        self.relu5 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.5)

        self.fc3 = nn.Linear(512, num_classes)
        self.logSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool1(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool2(x)

        x = x.view(-1, 128 * 10 * 10)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu5(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.logSoftmax(x)
        return x


class CNN2(nn.Module):
    def __init__(self, num_classes):
        super(CNN2, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3))      # 32@46*46
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))                 # 32@22*22

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3))     # 64@20*20
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3))    # 128@18*18
        self.relu3 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))                 # 128@9*9

        self.fc1 = nn.Linear(128 * 9 * 9, 1024)
        self.relu4 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.7)

        self.fc2 = nn.Linear(1024, 512)
        self.relu5 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.7)

        self.fc3 = nn.Linear(512, num_classes)
        self.logSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool2(x)

        x = x.view(-1, 128 * 9 * 9)
        x = self.fc1(x)
        x = self.relu4(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu5(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.logSoftmax(x)
        return x


class CNN3(nn.Module):
    def __init__(self, num_classes):
        super(CNN3, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=24, kernel_size=(3, 3))      # 24@46*46
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=24, out_channels=64, kernel_size=(3, 3))     # 64@44*44
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.25)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))                 # 64@22*22

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=96, kernel_size=(3, 3))     # 96@20*20
        self.relu3 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))                 # 96@10*10
        self.conv4 = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(3, 3))     # 96@8*8
        self.relu4 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.25)
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))                 # 96@4*4

        self.conv5 = nn.Conv2d(in_channels=96, out_channels=128, kernel_size=(3, 3))    # 128@2*2
        self.bn1 = nn.BatchNorm2d(128)
        self.relu5 = nn.ReLU()

        self.fc1 = nn.Linear(128 * 2 * 2, 1024)
        self.relu6 = nn.ReLU()
        self.fc2 = nn.Linear(1024, 1024)
        self.relu7 = nn.ReLU()
        self.dropout3 = nn.Dropout(p=0.25)

        self.fc3 = nn.Linear(1024, num_classes)
        self.logSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.dropout1(x)
        x = self.maxpool1(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool2(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.dropout2(x)
        x = self.maxpool3(x)

        x = self.conv5(x)
        x = self.bn1(x)
        x = self.relu5(x)

        x = x.view(-1, 128 * 2 * 2)
        x = self.fc1(x)
        x = self.relu6(x)
        x = self.fc2(x)
        x = self.relu7(x)
        x = self.dropout3(x)
        x = self.fc3(x)
        x = self.logSoftmax(x)
        return x


class CNN4(nn.Module):
    def __init__(self, num_classes):
        super(CNN4, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=24, kernel_size=(3, 3))      # 24@46*46
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=24, out_channels=64, kernel_size=(3, 3))     # 64@44*44
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.25)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))                 # 64@22*22

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=96, kernel_size=(3, 3))     # 96@20*20
        self.relu3 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))                 # 96@10*10
        self.conv4 = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(3, 3))     # 96@8*8
        self.relu4 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.5)
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))                 # 96@4*4

        self.conv5 = nn.Conv2d(in_channels=96, out_channels=128, kernel_size=(3, 3))    # 128@2*2
        self.bn1 = nn.BatchNorm2d(128)
        self.relu5 = nn.ReLU()

        self.fc1 = nn.Linear(128 * 2 * 2, 1024)
        self.relu6 = nn.ReLU()
        self.dropout3 = nn.Dropout(p=0.5)

        self.fc3 = nn.Linear(1024, num_classes)
        self.logSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.dropout1(x)
        x = self.maxpool1(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool2(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.dropout2(x)
        x = self.maxpool3(x)

        x = self.conv5(x)
        x = self.bn1(x)
        x = self.relu5(x)

        x = x.view(-1, 128 * 2 * 2)
        x = self.fc1(x)
        x = self.relu6(x)
        x = self.dropout3(x)
        x = self.fc3(x)
        x = self.logSoftmax(x)
        return x


class CNN5(nn.Module):
    def __init__(self, num_classes):
        super(CNN5, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 5))      # 32@44*44
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5))     # 64@40*40
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.25)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))                 # 64@20*20

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=96, kernel_size=(3, 3))     # 96@18*18
        self.bn3 = nn.BatchNorm2d(96)
        self.relu3 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.25)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))                 # 96@9*9

        self.conv4 = nn.Conv2d(in_channels=96, out_channels=128, kernel_size=(3, 3))    # 128@7*7
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU()
        self.dropout3 = nn.Dropout(p=0.25)

        self.fc1 = nn.Linear(128 * 7 * 7, 1024)
        self.relu5 = nn.ReLU()
        self.dropout4 = nn.Dropout(p=0.25)

        self.fc2 = nn.Linear(1024, num_classes)
        self.logSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout1(x)
        x = self.maxpool1(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.dropout2(x)
        x = self.maxpool2(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.dropout3(x)

        x = x.view(-1, 128 * 7 * 7)
        x = self.fc1(x)
        x = self.relu5(x)
        x = self.dropout4(x)
        x = self.fc2(x)
        x = self.logSoftmax(x)
        return x


class CNN6(nn.Module):
    def __init__(self, num_classes):
        super(CNN6, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 5))      # 32@44*44
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5))     # 64@40*40
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.25)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))                 # 64@20*20

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=96, kernel_size=(3, 3))     # 96@18*18
        self.bn3 = nn.BatchNorm2d(96)
        self.relu3 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.25)
        self.conv4 = nn.Conv2d(in_channels=96, out_channels=128, kernel_size=(3, 3))    # 128@16*16
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU()
        self.dropout3 = nn.Dropout(p=0.25)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))                 # 128@8*8

        self.fc1 = nn.Linear(128 * 8 * 8, 1024)
        self.relu5 = nn.ReLU()
        self.dropout4 = nn.Dropout(p=0.25)

        self.fc2 = nn.Linear(1024, num_classes)
        self.logSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout1(x)
        x = self.maxpool1(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.dropout2(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.dropout3(x)
        x = self.maxpool2(x)

        x = x.view(-1, 128 * 8 * 8)
        x = self.fc1(x)
        x = self.relu5(x)
        x = self.dropout4(x)
        x = self.fc2(x)
        x = self.logSoftmax(x)
        return x


class CNN7(nn.Module):  # BEST
    def __init__(self, num_classes):
        super(CNN7, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(5, 5))      # 32@44*44
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5))     # 64@40*40
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.25)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))                 # 64@20*20

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=96, kernel_size=(3, 3))     # 96@18*18
        self.bn3 = nn.BatchNorm2d(96)
        self.relu3 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.25)
        self.conv4 = nn.Conv2d(in_channels=96, out_channels=128, kernel_size=(3, 3))    # 128@16*16
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU()
        self.dropout3 = nn.Dropout(p=0.25)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))                 # 128@8*8

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3))   # 128@6*6
        self.bn5 = nn.BatchNorm2d(128)
        self.relu5 = nn.ReLU()
        self.dropout4 = nn.Dropout(p=0.25)
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))                 # 128@3*3

        self.fc1 = nn.Linear(128 * 3 * 3, 1024)
        self.relu6 = nn.ReLU()
        self.dropout5 = nn.Dropout(p=0.25)

        self.fc2 = nn.Linear(1024, num_classes)
        self.logSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout1(x)
        x = self.maxpool1(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.dropout2(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.dropout3(x)
        x = self.maxpool2(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        x = self.dropout4(x)
        x = self.maxpool3(x)

        x = x.view(-1, 128 * 3 * 3)
        x = self.fc1(x)
        x = self.relu6(x)
        x = self.dropout5(x)
        x = self.fc2(x)
        x = self.logSoftmax(x)
        return x


class CNN8(nn.Module):  # BEST
    def __init__(self, num_classes):
        super(CNN8, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=24, kernel_size=(5, 5))      # 24@44*44
        self.bn1 = nn.BatchNorm2d(24)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=24, out_channels=64, kernel_size=(5, 5))     # 64@40*40
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.25)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))                 # 64@20*20

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=96, kernel_size=(3, 3))     # 96@18*18
        self.bn3 = nn.BatchNorm2d(96)
        self.relu3 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.25)
        self.conv4 = nn.Conv2d(in_channels=96, out_channels=128, kernel_size=(3, 3))    # 128@16*16
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU()
        self.dropout3 = nn.Dropout(p=0.25)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))                 # 128@8*8

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3))   # 128@6*6
        self.bn5 = nn.BatchNorm2d(128)
        self.relu5 = nn.ReLU()
        self.dropout4 = nn.Dropout(p=0.25)
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))                 # 128@3*3

        self.fc1 = nn.Linear(128 * 3 * 3, 1024)
        self.relu6 = nn.ReLU()
        self.dropout5 = nn.Dropout(p=0.25)

        self.fc2 = nn.Linear(1024, num_classes)
        self.logSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout1(x)
        x = self.maxpool1(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.dropout2(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.dropout3(x)
        x = self.maxpool2(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        x = self.dropout4(x)
        x = self.maxpool3(x)

        x = x.view(-1, 128 * 3 * 3)
        x = self.fc1(x)
        x = self.relu6(x)
        x = self.dropout5(x)
        x = self.fc2(x)
        x = self.logSoftmax(x)
        return x