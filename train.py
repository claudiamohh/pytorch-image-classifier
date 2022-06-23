'''Convolutional Neural Network'''
import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim

import torchvision 
import torchvision.transforms as transforms 

import os
import argparse

from models.LinearClassifier import Linear
from models.CNNClassifier import CNN
from datasets.CIFAR10 import CIFAR10Dataset
from datasets.MNIST import MNISTDataset

parser = argparse.ArgumentParser(description='Convolutional Neural Network Training')
parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
parser.add_argument('--epoch', default=10, type=int, help='Number of epoch')
parser.add_argument('--model')
parser.add_argument('--dataset')
args = parser.parse_args()

#Checking if GPU is available
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(device)


def train (model, optimizer, criterion, trainloader, epochs):
    '''Training the model using the variables: model, optimizer, criterion, trainloader, number of epochs'''
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, data in enumerate(trainloader, 0):

            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad() #zero the gradients of the model parameters 

            #forward + backward + optimize 
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            #performing the backpropogation step and updating the weights of the model parameters
            loss.backward()
            optimizer.step()

            if batch_idx >= 2:
                break

            running_loss += loss.item() #loss item of each batch 
        print(f'Loss of Epoch: {epoch} is {running_loss/len(trainloader)}')

    print('Finished Training')
    return model 

def evaluate_overall(model, testloader):
    '''To calculate the overall accuracy of the model'''
    correct = 0
    total = 0

    #prediction for testing set
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data[0].to(device), data[1].to(device)
            outputs = model(inputs)     #calculate outputs by running inputs through the network
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the test images: {100 * correct // total} %')

def main():
    dataset = MNISTDataset() if args.dataset == "mnist" else CIFAR10Dataset()
    channels = dataset.channels()
    size = dataset.size()
    train_loader, test_loader = dataset.get_loaders()
    criterion = nn.CrossEntropyLoss()       #Loss function to calculate the difference between input and target
    model = CNN(channels, use_mnist=(args.dataset == "mnist")) if args.model == "cnn" else Linear(channels, size)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)      #Adam optimizer 

    model = train(model, optimizer, criterion, train_loader, args.epoch)
    evaluate_overall(model, test_loader)

if __name__ == '__main__':
    main()
