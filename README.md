# Convolutional Neural Network

A PyTorch implementation for training a large sized convolutional neural network and a linear classifier model on CIFAR-10 dataset and MNIST dataset.

There are two training scripts: train.py and lightning_train.py. Pytorch lightning is used in lightning_train.py, introducing simplicity to the codes in train.py. It increases efficiency, making it reproducibility and decreases the speed. In lightning_train.py, it uses the same models and datasets as train.py. 

CIFAR-10 dataset consists of 60,000 coloured images, 50,000 images form the training data and the remaining 10,000 forming the test data. Each image has a dimension of 32 * 32 pixels. CIFAR-10 has 10 classes, each classes having 6000 images.

MNIST dataset consists of handwritten digits between 0 and 9. It has 70,000 handwritten digits, 60,000 digits for the train set and 10,000 for the test set. Each digit is stored in a grayscale image with a size of 28 * 28 pixels.

## Model
A large sized convolutional neural network and a linear classifier are created to train and test the accuracy of the images and digits in CIFAR-10 and MNIST dataset respectively.

1. CNN Classifier 
```
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
```

The large sized CNN model consists of 6 convolution layers and 3 fully connected layers. The first convolution layer takes in the dataset's channel (CIFAR-10=3, MNIST=1) and a kernel size of 3 * 3. The output of this convolution layer is set to 32 channels, which means it will extract 32 feature maps using 32 kernels. Padding size is set to 1 so that input and output dimensions are the same. It will then go through ReLU activation followed by a max-pooling layer with kernel size of 2 and stride 2.

It will then half the spatial dimensions by passing through a maxpool layer over a 2 * 2 window and stride 2. The final convolution layer will produce a channeled output of 256 with the spatial dimensions now being 4 * 4.

Finally, these feature maps will be flattened and passed through the fully connected layers to get an output of 10 and the accuracy.

CNN() is used in train.py while LightningCNN() is used in lightning_train.py to train the CNN Classifier model with the above parameters.

2. Linear Classifier

```
self.fc1 = nn.Linear(
            channels * size * size, 120
        )
self.fc2 = nn.Linear(120, 84)
self.fc3 = nn.Linear(84, 10)
```

The Linear Classifier model consists of 3 fully connected layers. Firstly, the dataset will be flattened to 1-dimension. The first fully connected layer will then take in the flattened input and apply ReLU activation function to produce an output of 120 channels. This continues to fc3 where it has an output channel of 10.

Linear() is used in train.py while LightningLinear() is used in lightning_train.py to train the Linear Classifier model with the above parameters. 

## Training
<ins>Pytorch</ins>

The accuracy for CNN Model:
1. with CIFAR10 dataset ~79%
2. with MNIST dataset ~99%

The accuracy for Linear Model:
1. with CIFAR10 dataset ~52%
2. with MNIST dataset ~97%

<ins>Pytorch Lightning</ins>

The accuracy for LightningCNN Model:
1. with LightningCIFAR10 dataset ~78%
2. with LightningMNIST dataset ~99%

The accuracy for LightningLinearModel: 
1. with LightningCIFAR10 dataset ~51%
2. with Lightning MNIST dataset ~97%

All models use Adam Optimizer and Cross Entropy Loss to calculate the loss between input and target.

## Requirements
1. torch
2. torchvision
3. tqdm
4. pytorch-lightning 

To install requirements:
```
$ pip install -r requirements.txt
```

To install and activate virtual environment:
```
$ python -m venv env

# Linux
source env/bin/activate

# Windows
source env/Scripts/activate
```

## Execution
Both scripts use argsparse to write user-friendly command-line interfaces. Hence in terminal, type the following command to start the training process:

```
$ python train.py

# with pytorch lightning
$ python lightning_train.py
```

By default, it trains the Linear Model with CIFAR10 dataset in train.py and LightningLinear Model with LightningCIFAR10 dataset in lightning_train.py.

There are 4 variables that can be altered: learning rate, number of epoch, model and dataset. Learning rate (by default: 0.001) and epoch (by default: 10) can be changed to a user's preferred value.

To use CNN Classifier model, type the following command:
```
$ python train.py --model cnn

# with pytorch lightning
$ python lightning_train.py --model cnn
```

To use MNIST dataset. type the following command:

```
$ python train.py --dataset mnist

# with pytorch lightning
$ python lightning_train.py --dataset mnist
```

Hence, to train a CNN Classifier model with MNIST dataset using 30 epochs and learning rate of 0.01 will be:

```
$ python train.py --model cnn --dataset mnist --epoch 30 --lr 0.01

# with pytorch lightning 
$ python lightning_train.py --model cnn --dataset mnist --epoch 30 --lr 0.01
```

