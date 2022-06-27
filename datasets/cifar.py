import os
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, random_split


class CIFAR10Dataset:
    def __init__(self, batch_size=256):
        super().__init__()

        self.batch_size = batch_size

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        self.trainset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=self.transform
        )

        self.trainloader = torch.utils.data.DataLoader(
            self.trainset, batch_size=self.batch_size, shuffle=True, num_workers=2
        )

        self.testset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=self.transform
        )

        self.testloader = torch.utils.data.DataLoader(
            self.testset, batch_size=self.batch_size, shuffle=False, num_workers=2
        )

    def channels(self):
        # image size = (3 * 32 * 32)
        return 3

    def size(self):
        # image size = (3 * 32 * 32)
        return 32

    def get_loaders(self):
        return (self.trainloader, self.testloader)

#Lightning CIFAR10 Dataset
class LightningCIFAR10Dataset:
    def __init__(self, batch_size=256):
        super().__init__()

        self.batch_size = batch_size

        self.input_size = 32 

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        self.train_dataset = CIFAR10(
                root=os.getcwd(), train=True, transform=self.transform, download=True
        )

        self.test_dataset = CIFAR10(
                root=os.getcwd(), train=False, transform=self.transform, download=True
        )

        self.train_set_size = int(len(self.train_dataset) * 0.8)
        self.valid_set_size = len(self.train_dataset) - self.train_set_size

        self.seed = torch.Generator().manual_seed(42)
        self.train_dataset, self.valid_dataset = random_split(
            self.train_dataset, [self.train_set_size, self.valid_set_size], generator=self.seed
        )
    
    def channels(self):
        # image size = (3 * 32 * 32)
        return 3

    def size(self):
        # image size = (3 * 32 * 32)
        return 32

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,)

    def valid_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
