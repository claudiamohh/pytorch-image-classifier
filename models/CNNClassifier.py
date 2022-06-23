''' Large CNN Model with 6 convolution layers, pooling layers and 3 fully connected layers.
First convolution layer takes in a channel of dimension 3 since the images are RGB, kernel size is 3*3. The output of this convolution layer is set to 32 channels which means it will extract 32 feature maps using 32 kernels. Padding size is set to 1 so that input and output dimensions are the same. Then it goes through ReLu activation followed by a max-pooling layer with kernel size of 2 and stride 2.'''

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from datasets.MNIST import MNISTDataset

use_mnist = MNISTDataset()

class CNN(nn.Module):
    def __init__(self, channels, use_mnist=False):
        super().__init__()
        self.channels = channels
        self.use_mnist = use_mnist

        self.conv1 = nn.Conv2d(self.channels, 32, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 3, stride = 1, padding = 1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size = 3, stride = 1, padding = 1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size = 3, stride = 1, padding = 1)
        self.pool = nn.MaxPool2d(2, 2)

        if self.use_mnist:
            self.fc0 = nn.Linear(2304, 256 * 4 * 4)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool(F.relu(self.conv4(x)))
        x = F.relu(self.conv5(x))
        x = self.pool(F.relu(self.conv6(x)))
        x = torch.flatten(x, 1)

        if self.use_mnist:
            x = F.relu(self.fc0(x))

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
