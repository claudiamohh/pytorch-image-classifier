class CIFAR10Dataset():
    def __init__(self):
        super().__init__()

        self.transform = transforms.Compose(
                [transforms.ToTensor(),
                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        self.trainset = torchvision.datasets.CIFAR10(
                root='./data', train=True, download=True, transform=transform)

        self.trainloader = torch.utils.data.DataLoader(
                trainset, batch_size=batch_size, shuffle=True, num_workers=2)


        self.testset = torchvision.datasets.CIFAR10(
                root='./data', train=False, download=True, transform=transform)

        self.testloader = torch.utils.data.DataLoader(
                testset, batch_size=batch_size, shuffle=False, num_workers=2)

    def get_loaders():
        return (self.trainloader, self.testloader)
