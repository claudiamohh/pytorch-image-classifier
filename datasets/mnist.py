""" MNIST Dataset """
import os
import torch
import pytorch_lightning as pl
import torchvision.transforms as transforms
from torchvision import datasets
from torchvision.datasets import MNIST
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


class LightningMNISTDataset(pl.LightningDataModule):
    def __init__(self, batch_size=256, input_size=28):
        super().__init__()

        self.batch_size = batch_size
        self.input_size = input_size
        self.transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5), (0.5))]
        )

    def setup(self, stage=None):
        self.train_dataset = MNIST(
            root=os.getcwd(), train=True, transform=self.transform, download=True
        )
        self.test_dataset = MNIST(
            root=os.getcwd(), train=False, transform=self.transform, download=True
        )

        self.train_set_size = int(len(self.train_dataset) * 0.8)
        self.valid_set_size = len(self.train_dataset) - self.train_set_size

        self.seed = torch.Generator().manual_seed(42)
        self.train_dataset, self.valid_dataset = random_split(
            self.train_dataset,
            [self.train_set_size, self.valid_set_size],
            generator=self.seed,
        )

    def channels(self):
        # image size = (1 * 28 * 28)
        return 1

    def size(self):
        # image size = (1 * 28 * 28)
        return 28

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(self.valid_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
