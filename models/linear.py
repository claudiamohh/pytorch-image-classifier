import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import pytorch_lightning as pl

class Linear(nn.Module):
    def __init__(self, channels, size):
        super().__init__()
        self.fc1 = nn.Linear(
            channels * size * size, 120
        )  # input has to be image size
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class LightningLinear(pl.LightningModule):
    def __init__(self, channels, size):
        super().__init__()
        self.model = Linear(channels, size)

    def forward(self,x):
        return self.model(x)
    
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
