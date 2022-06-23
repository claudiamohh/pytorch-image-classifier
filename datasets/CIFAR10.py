import torch
import torchvision
import torchvision.transforms as transforms

class CIFAR10Dataset():
    def __init__(self, batch_size=256):
        super().__init__()

        self.batch_size = batch_size

        self.transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.trainset = torchvision.datasets.CIFAR10(
                root='./data', train=True, download=True, transform=self.transform)

        self.trainloader = torch.utils.data.DataLoader(
                self.trainset, batch_size=self.batch_size, shuffle=True, num_workers=2)


        self.testset = torchvision.datasets.CIFAR10(
                root='./data', train=False, download=True, transform=self.transform)

        self.testloader = torch.utils.data.DataLoader(
                self.testset, batch_size=self.batch_size, shuffle=False, num_workers=2)

#image size = (3 * 32 * 32)
    def channels(self):
        return 3
    
    def size(self):
        return 32

    def get_loaders(self):
        return (self.trainloader, self.testloader)
