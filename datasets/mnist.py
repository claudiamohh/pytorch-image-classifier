""" MNIST Dataset """
import os
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.datasets import MNIST      
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, random_split

class MNISTDataset:
    def __init__(self, batch_size=256):
        super().__init__()

        self.batch_size = batch_size

        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5), (0.5))]
        )

        self.trainset = datasets.MNIST(
            root="data", train=True, transform=self.transform, download=True
        )

        self.trainloader = torch.utils.data.DataLoader(
            self.trainset, batch_size=self.batch_size, shuffle=True, num_workers=1
        )

        self.testset = datasets.MNIST(
            root="data", train=False, transform=self.transform
        )

        self.testloader = torch.utils.data.DataLoader(
            self.testset, batch_size=self.batch_size, shuffle=True, num_workers=1
        )

    def channels(self):
        # image size = (1 * 28 * 28)
        return 1

    def size(self):
        # image size = (1 * 28 * 28)
        return 28

    def get_loaders(self):
        return (self.trainloader, self.testloader)


#Lightning MNIST Dataset

class LightningMNISTDataset:
    def __init__(self, batch_size=256):
        super().__init__()

        self.batch_size = batch_size

        self.input_size = 224

        self.train_transform = transforms.Compose(
                [
                    transforms.RandomResizedCrop(self.input_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]
        )
        
        self.test_transform = transforms.Compose(
                [
                    transforms.Resize(self.input_size),
                    transforms.CenterCrop(self.input_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]
        )
        
        self.train_dataset = MNIST(
                root=os.getcwd(), train=True, transform=self.train_transform, download=True
        )

        self.test_dataset = MNIST(
                root=os.getcwd(), train=False, transform=self.test_transform, download=True
        )

        self.train_set_size = int(len(self.train_dataset) * 0.8)
        self.valid_set_size = len(self.train_dataset) - self.train_set_size

        self.seed = torch.Generator().manual_seed(42)
        self.train_dataset, self.valid_dataset = random_split(
            self.train_dataset, [self.train_set_size, self.valid_set_size], generator=self.seed
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,)

    def valid_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
