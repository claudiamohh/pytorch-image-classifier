"""Training script to train linear/CNN model for CIFAR10/MNIST dataset with pytorch lightning"""
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

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
    use_mnist = args.dataset == "mnist"

    model = (
           LightningCNN(channels, args.lr, use_mnist=use_mnist)
           if args.model == "cnn"
           else LightningLinear(channels, size, args.lr)
    )
    
    gpu_counts = min(1, torch.cuda.device_count())
    accelerator= "cpu" if gpu_counts <= 0 else "gpu"
    early_stop_callback = EarlyStopping(monitor="valid_loss", patience=5, verbose=False, mode="min")

    trainer = pl.Trainer(accelerator=accelerator,
                     devices=gpu_counts,
                     max_epochs=args.epoch,
                     callbacks=[early_stop_callback],
                     #logger=TensorBoardLogger("claudia_lightning_logs/", name="claudia_lightning_model")
                     )

    trainer.fit(model=model,
                datamodule=dataset,
                )

    trainer.test(model=model, datamodule=dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description=" Convolutional Neural Network Training with Pytorch Lightning"
    )
    parser.add_argument("--lr", default=0.001, type=float, help="Learning rate")
    parser.add_argument("--epoch", default=10, type=int, help="Number of epoch")
    parser.add_argument("--model", default="linear", help="Type of model to be trained")
    parser.add_argument("--dataset", default="cifar10", help="Dataset to be used" )
    args = parser.parse_args()

    main(args)
