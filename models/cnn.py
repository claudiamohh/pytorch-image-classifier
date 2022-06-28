import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import pytorch_lightning as pl

class CNN(nn.Module):
    """Large CNN Model with 6 convolution layers, pooling layers and 3 fully
    connected layers.  First convolution layer takes in a channel of dimension 3
    since the images are RGB, kernel size is 3*3. The output of this convolution
    layer is set to 32 channels which means it will extract 32 feature maps using
    32 kernels. Padding size is set to 1 so that input and output dimensions are
    the same. Then it goes through ReLu activation followed by a max-pooling layer
    with kernel size of 2 and stride 2.
    """

    def __init__(self, channels, use_mnist=False):
        super().__init__()
        self.channels = channels
        self.use_mnist = use_mnist

        self.conv1 = nn.Conv2d(self.channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        if self.use_mnist:
            self.fc0 = nn.Linear(2304, 256 * 4 * 4)

        self.fc1 = nn.Linear(256 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
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


class LightningCNN(pl.LightningModule):
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

        self.fc1 = nn.Linear(256 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 10)

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
    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self.forward(x)

        loss = F.cross_entropy(logits, y.long())

        preds = torch.argmax(logits, -1)
        acc = accuracy(preds, y)
        f1 = f1_score(preds, y, num_classes=10)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True) # show metrics in progress bar
            self.log(f"{stage}_acc", acc, prog_bar=True)
            self.log(f"{stage}_f1", f1, prog_bar=True)
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "valid")
    
    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def predict_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        return logits 

    def configure_optimizers(self, lr):
        self.lr = lr
        return torch.optim.Adam(self.parameters(), lr=self.lr)
