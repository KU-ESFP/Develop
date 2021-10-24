import torch.nn as nn
import torch.nn.functional as F


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

        x = x.view(-1, 50*9*9)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.logSoftmax(x)
        return x