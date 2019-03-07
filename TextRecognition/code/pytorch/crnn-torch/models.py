import torch
import torch.nn as nn

class CRNN(nn.Modules):
    def __init__(self):
        super(CRRN, self).__init__()

        self.features = nn.Sequential(
                # in_channels, out_channels, kernel_size, padding
                nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1),
                nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1),
                nn.MaxPool2d(kernel_size=1, stride=2),
                nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=1),
                nn.BatchNorm2d(512),
                nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=1),
                nn.BatchNorm2d(512),
                nn.MaxPool2d(kernel_size=1, stride=2),
                nn.Conv2d(512, 512, kernel_size=2, stride=1),
                )






