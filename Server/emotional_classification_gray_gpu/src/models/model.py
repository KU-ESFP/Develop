import torch.nn as nn
import torch.nn.functional as F

'''
    [Convolution] - (1, 48, 48)
    W: 너비
    K: kernel_size
    P: padding
    S: stride
    = (W - K + P * 2) / S + 1
'''

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
