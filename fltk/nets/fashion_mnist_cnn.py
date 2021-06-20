from typing import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import dropout

class FashionMNISTCNN(nn.Module):

    def __init__(self):
        super(FashionMNISTCNN, self).__init__()

        self.layer1 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(1, 16, kernel_size=5, padding=2)),
            ('batch', nn.BatchNorm2d(16)),
            ('relu', nn.ReLU()),
            ('mpool', nn.MaxPool2d(2)),
            ('drop', nn.Dropout(p=0))
        ]))
        self.layer2 = nn.Sequential(OrderedDict([
            ('conv', nn.Conv2d(16, 32, kernel_size=5, padding=2)),
            ('batch', nn.BatchNorm2d(32)),
            ('relu', nn.ReLU()),
            ('mpool', nn.MaxPool2d(2)),
            ('drop', nn.Dropout(p=0))
        ]))

        self.fc = nn.Linear(7*7*32, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)

        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x
