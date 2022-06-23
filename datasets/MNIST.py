''' MNIST Dataset '''

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.transforms import ToTensor

class MNISTDataset():
    def __init__(self, batch_size = 256):
        super().__init__()

        self.batch_size = batch_size

        self.transform = transforms.Compose(
                [transforms.ToTensor(),
                  transforms.Normalize((0.5), (0.5))])

        self.trainset = datasets.MNIST(
                root = 'data', train = True, transform = self.transform, download = True)

        self.trainloader = torch.utils.data.DataLoader(
                self.trainset, batch_size = self.batch_size, shuffle = True, num_workers = 1)

        
        self.testset = datasets.MNIST(
                root = 'data', train = False, transform = self.transform)

        self.testloader = torch.utils.data.DataLoader(
                self.testset, batch_size = self.batch_size, shuffle = True, num_workers = 1)

    def channels(self):
        # image size = (1 * 28 * 28)
        return 1

    def size(self):
        # image size = (1 * 28 * 28)
        return 28

    def get_loaders(self):
        return (self.trainloader, self.testloader)
