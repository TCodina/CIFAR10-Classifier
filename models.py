"""
This file contains all deep-learning models tested in the CIFAR-10 classification survey.
TODO: Many classes can be merged upon adding additional arguments.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Linear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(32 * 32 * 3, 10)

    def forward(self, x):
        x = torch.flatten(x, 1) # Flatten all dimensions except batch
        x = F.softmax(self.fc(x), dim=1)
        return x

class Dense(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32 * 32 * 3, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x


class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x), dim=1)
        return x


class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()

        ch = [3, 96, 256, 384, 256]
        fc_inout = 4096
        self.conv_layers = nn.Sequential(
            nn.Conv2d(ch[0], ch[1], kernel_size=11, stride=4, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(ch[1], ch[2], kernel_size=5, stride=1, padding=2, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(ch[2], ch[3], kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch[3], ch[3], kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch[3], ch[4], kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(ch[4] * 6 * 6, fc_inout, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(fc_inout, fc_inout, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(fc_inout, 10),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        x = self.fc_layers(x)
        return x


class AlexNetNorm(nn.Module):
    def __init__(self, p=0.5):
      super().__init__()

      ch = [3, 96, 256, 384, 256]
      fc_inout = 4096
      self.conv_layers = nn.Sequential(
          nn.Conv2d(ch[0], ch[1], kernel_size=11, stride=4, padding=0, bias=False),
          nn.BatchNorm2d(ch[1]),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=3, stride=2),
          nn.Conv2d(ch[1], ch[2], kernel_size=5, stride=1, padding=2, bias=False),
          nn.BatchNorm2d(ch[2]),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=3, stride=2),
          nn.Conv2d(ch[2], ch[3], kernel_size=3, stride=1, padding=1, bias=False),
          nn.BatchNorm2d(ch[3]),
          nn.ReLU(inplace=True),
          nn.Conv2d(ch[3], ch[3], kernel_size=3, stride=1, padding=1, bias=False),
          nn.BatchNorm2d(ch[3]),
          nn.ReLU(inplace=True),
          nn.Conv2d(ch[3], ch[4], kernel_size=3, stride=1, padding=1, bias=False),
          nn.BatchNorm2d(ch[4]),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=3, stride=2),
      )

      self.classifier = self.classifier = nn.Sequential(
            nn.Linear(ch[4] * 6 * 6, fc_inout, bias=False),
            nn.BatchNorm1d(fc_inout),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(fc_inout, fc_inout, bias=False),
            nn.BatchNorm1d(fc_inout),
            nn.ReLU(inplace=True),
            nn.Dropout(p),
            nn.Linear(fc_inout, 10, bias=True),
            nn.Softmax(dim=1)
        )

      self.initialize_parameters()

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


    def initialize_parameters(self):
        '''
        Initialize parameters
        '''
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')  # He et al.
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)  # Fixed bias
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):  # Standard BatchNorm initialization
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class AlexNet1Fc(nn.Module):
    def __init__(self):
      super().__init__()

      ch = [3, 96, 256, 384, 256]
      self.conv_layers = nn.Sequential(
          nn.Conv2d(ch[0], ch[1], kernel_size=11, stride=4, padding=0, bias=False),
          nn.BatchNorm2d(ch[1]),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=3, stride=2),
          nn.Conv2d(ch[1], ch[2], kernel_size=5, stride=1, padding=2, bias=False),
          nn.BatchNorm2d(ch[2]),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=3, stride=2),
          nn.Conv2d(ch[2], ch[3], kernel_size=3, stride=1, padding=1, bias=False),
          nn.BatchNorm2d(ch[3]),
          nn.ReLU(inplace=True),
          nn.Conv2d(ch[3], ch[3], kernel_size=3, stride=1, padding=1, bias=False),
          nn.BatchNorm2d(ch[3]),
          nn.ReLU(inplace=True),
          nn.Conv2d(ch[3], ch[4], kernel_size=3, stride=1, padding=1, bias=False),
          nn.BatchNorm2d(ch[4]),
          nn.ReLU(inplace=True),
          nn.MaxPool2d(kernel_size=3, stride=2),
      )

      self.classifier = self.classifier = nn.Sequential(
            nn.Linear(ch[4] * 6 * 6, 10, bias=True),
            nn.Softmax(dim=1)
        )

      self.initialize_parameters()

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


    def initialize_parameters(self):
        '''
        Initialize parameters
        '''
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')  # He et al.
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)  # fixed bias
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):  # standard BatchNorm initialization
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

class VGG16(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()

        conv_channels = [64, 64, 'p', 128, 128, 'p', 256, 256, 256, 'p', 512, 512, 512, 'p', 512, 512, 512, 'p']
        self.conv_layers = self.build_conv_layers(conv_channels)
        self.fc = nn.Sequential(
            nn.Linear(conv_channels[-2] * 7 * 7, 4096, bias=False),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(4096, 4096, bias=False),
            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(4096, 10),
            nn.Softmax(dim=1)
        )

        self.initialize_parameters()

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def build_conv_layers(self, ch):
      ch_in = 3
      layers = []
      for ch_out in ch:
        if type(ch_out) == int:
          layers += [nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=False),
                     nn.BatchNorm2d(ch_out),
                     nn.ReLU()]
          ch_in = ch_out
        elif ch_out == 'p':
          layers += [nn.MaxPool2d(kernel_size=2, stride=2)]


      return nn.Sequential(*layers)

    def initialize_parameters(self):
        '''
        Initialize parameters
        '''
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')  # He et al.
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)  # fixed bias
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):  # standard BatchNorm initialization
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class ResNet50Wrapper(nn.Module):
  def __init__(self):
    super().__init__()

    self.resnet50 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', weights='ResNet50_Weights.IMAGENET1K_V1')
    
    # Change fully-connected layer
    self.resnet50.fc = nn.Sequential(nn.Linear(in_features=2048, out_features=10), nn.Softmax(dim=1))

  def forward(self, x):
    x = self.resnet50(x)
    return x

class DenseNet121Wrapper(nn.Module):
  def __init__(self, drop_rate=0.0):
    super().__init__()
    
    self.drop_rate = drop_rate  # Dropout
    self.densenet121 = torch.hub.load('pytorch/vision:v0.10.0', 'densenet121', 
                                      weights='DenseNet121_Weights.IMAGENET1K_V1',
                                     drop_rate=self.drop_rate)
    
    self.densenet121.classifier = nn.Sequential(nn.Linear(in_features=1024, out_features=10, bias=True),
                                    nn.Softmax(dim=1))

  def forward(self, x):
    x = self.densenet121(x)
    return x
  
