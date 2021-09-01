import torch
import torch.nn as nn


class Conv2dSame(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 bias=True, padding_layer=torch.nn.ReflectionPad2d):
        """ It only support square kernels and stride=1, dilation=1, groups=1. """
        super().__init__()
        ka = kernel_size // 2
        kb = ka - 1 if kernel_size % 2 == 0 else ka
        self.net = torch.nn.Sequential(
            padding_layer((ka, kb, ka, kb)),
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, bias=bias)
        )

    def forward(self, x):
        return self.net(x)


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.sep = nn.Sequential(   # separable
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        )

    def forward(self, x):
        x = self.sep(x)
        return x


class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.layer_1 = nn.Sequential(
            Conv2dSame(in_channels=3, out_channels=16, kernel_size=7),
            nn.BatchNorm2d(16),
            Conv2dSame(in_channels=16, out_channels=16, kernel_size=7),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2),
            nn.Dropout(0.5)
        )
        self.layer_2 = nn.Sequential(
            Conv2dSame(in_channels=16, out_channels=32, kernel_size=5),
            nn.BatchNorm2d(32),
            Conv2dSame(in_channels=32, out_channels=32, kernel_size=5),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2),
            nn.Dropout(0.5)
        )
        self.layer_3 = nn.Sequential(
            Conv2dSame(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            Conv2dSame(in_channels=64, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2),
            nn.Dropout(0.5)
        )
        self.layer_4 = nn.Sequential(
            Conv2dSame(in_channels=64, out_channels=128, kernel_size=3),
            nn.BatchNorm2d(128),
            Conv2dSame(in_channels=128, out_channels=128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=2),
            nn.Dropout(0.5)
        )
        self.layer_5 = nn.Sequential(
            Conv2dSame(in_channels=128, out_channels=256, kernel_size=3),
            nn.BatchNorm2d(256),
            Conv2dSame(in_channels=256, out_channels=num_classes, kernel_size=3),
            nn.AdaptiveAvgPool2d(output_size=num_classes)
        )
        self.fc1 = nn.Linear(num_classes * num_classes * num_classes, num_classes)

    def forward(self, x):
        out = self.layer_1(x)
        out = self.layer_2(out)
        out = self.layer_3(out)
        out = self.layer_4(out)
        out = self.layer_5(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)

        return out


class MiniXception(nn.Module):
    def __init__(self, num_classes):
        super(MiniXception, self).__init__()

        # base
        self.feature = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )

        # module 1
        self.residual1 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=1, stride=2, padding=0),
            nn.BatchNorm2d(16)
        )

        self.block1 = nn.Sequential(
            SeparableConv2d(in_channels=8, out_channels=16),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            SeparableConv2d(in_channels=16, out_channels=16),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # module 2
        self.residual2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=1, stride=2, padding=0),
            nn.BatchNorm2d(32)
        )

        self.block2 = nn.Sequential(
            SeparableConv2d(in_channels=16, out_channels=32),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            SeparableConv2d(in_channels=32, out_channels=32),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # module 3
        self.residual3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1, stride=2, padding=0),
            nn.BatchNorm2d(64)
        )

        self.block3 = nn.Sequential(
            SeparableConv2d(in_channels=32, out_channels=64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            SeparableConv2d(in_channels=64, out_channels=64),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # module 4
        self.residual4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1, stride=2, padding=0),
            nn.BatchNorm2d(128)
        )

        self.block4 = nn.Sequential(
            SeparableConv2d(in_channels=64, out_channels=128),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            SeparableConv2d(in_channels=128, out_channels=128),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.block5 = nn.Sequential(
            Conv2dSame(in_channels=128, out_channels=num_classes, kernel_size=3),
            nn.AdaptiveAvgPool2d(output_size=num_classes)
        )
        self.fc1 = nn.Linear(num_classes * num_classes * num_classes, num_classes)

    def forward(self, x):
        out = self.feature(x)

        out = self.residual1(out) + self.block1(out)
        out = self.residual2(out) + self.block2(out)
        out = self.residual3(out) + self.block3(out)
        out = self.residual4(out) + self.block4(out)

        out = self.block5(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)

        return out


