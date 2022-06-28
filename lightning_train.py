"""Training script to train linear/CNN model for CIFAR10/MNIST dataset with pytorch lightning"""
import os
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from torchmetrics.functional import accuracy, f1_score
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelSummary

import argparse
from models.linear import LightningLinear
from models.cnn import LightningCNN
from datasets.cifar import LightningCIFAR10Dataset
from datasets.mnist import LightningMNISTDataset


def main(args):

    print("\n")
    print("Arguments".center(50, "-"))
    for key, value in vars(args).items():
        print(f" {key}: {value}")

    print("Arguments".center(50, "-"))
    print("\n")

    dataset = LightningMNISTDataset() if args.dataset == "mnist" else LightningCIFAR10Dataset()
    channels = dataset.channels()
    size = dataset.size()
    
    train_dataloaders = dataset.train_dataloader()
    val_dataloaders = dataset.valid_dataloader()
    test_dataloader = dataset.test_dataloader()
    
    use_mnist = args.dataset == "mnist"

    model = (
           LightningCNN(channels, use_mnist=use_mnist) 
           if args.model == "cnn"
           else LightningLinear(channels, size)
    )
    
    learning_rate = LightningCNN(channels).configure_optimizers(lr=args.lr)
    
    trainer = pl.Trainer(accelerator='cpu',
                     devices=1,
                     max_epochs=args.epoch,
                     callbacks=[early_stop_callback],
                     #logger=TensorBoardLogger("claudia_lightning_logs/", name="claudia_lightning_model")
                     )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description=" Convolutional Neural Network Training with Pytorch Lightning"
    )
    parser.add_argument("--lr", default=0.001, type=float, help="Learning rate")
    parser.add_argument("--epoch", default=10, type=int, help="Number of epoch")
    parser.add_argument("--model")
    parser.add_argument("--dataset")
    args = parser.parse_args()

    main(args)


early_stop_callback = EarlyStopping(monitor="valid_loss", patience=5, verbose=False, mode="min")


trainer.fit(model=LightningCNN(channels, use_mnist=use_mnist),
            train_dataloaders=train_dataloader,
            val_dataloaders=valid_dataloader,
            )

trainer.test(model, test_dataloader)
