# Coby Penso, 208254128
import matplotlib.pyplot as plt
import numpy as np
from numpy.testing._private.utils import break_cycles
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.models.mnasnet import _load_pretrained
import torchvision.transforms as transforms
import torch.optim as optim
from torchvision import models


class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim, device):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        if device == 'cuda':
            self.cuda()
    def forward(self, x):
        x = x.view(x.shape[0], -1)
        outputs = self.linear(x)
        return outputs

class FC3_Net(nn.Module):
    def __init__(self, input_dim, output_dim, device):
        super(FC3_Net, self).__init__()

        self.model = nn.Sequential(
            #First layer
            nn.Linear(input_dim, 100, bias=True),
            nn.ReLU(),
            #Second layer
            nn.Linear(100, 100, bias=True),
            nn.BatchNorm1d(100),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            #Third layer
            nn.Linear(100, 100, bias=True),
            nn.BatchNorm1d(100),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            #Fourth layer
            nn.Linear(100, 100, bias=True),
            nn.BatchNorm1d(100),
            nn.Dropout(p=0.5),
            nn.ReLU(),
            #Last later
            nn.Linear(100, output_dim),
        )
        if device == 'cuda':
            self.cuda()
            
    def forward(self, x):
        
        x = x.view(x.shape[0], -1)
        return self.model(x)

class CNN_Net(nn.Module):
    def __init__(self, input_dim, output_dim, device):
        super(CNN_Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)

        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(16 * 21 * 21, output_dim)
        if device == 'cuda':
            self.cuda()
            
    def forward(self, x, evalMode=False):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(-1, 16 * 21 * 21)
        return self.fc(x)

class PreTrained_ResNet18(nn.Module):
    def __init__(self, hidden_sizes, output_dim, feature_extract, device):
        super(PreTrained_ResNet18, self).__init__()
        self.feature_extractor = models.resnet18(pretrained=True)
        self.hidden_sizes = hidden_sizes
        self.layers = []

        for i in range(len(self.hidden_sizes) - 1):
            self.layers.append(nn.Linear(self.hidden_sizes[i], self.hidden_sizes[i+1]))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(0.5))

        self.layers.append(nn.Linear(hidden_sizes[-1], output_dim))
        self.net = nn.Sequential(*self.layers)
        
        if feature_extract:
            for param in self.feature_extractor.parameters():
                param.require_grad = False
                
        num_ftrs = self.feature_extractor.fc.in_features
        self.feature_extractor.fc = nn.Linear(num_ftrs, output_dim)
        
        if device == 'cuda':
            self.cuda()
            
    def forward(self, x, evalMode=False):
        output = self.feature_extractor(x)
        return output

def init_weights(m):
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight.data)