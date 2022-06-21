import torch
import torchvision 
import torchvision.transforms as transforms 
import torch.nn as nn
import torch.nn.functional as F 
import torach.optim as optim
import os
import argparse
from models.cnn import CNNClassifier
from models.cnn import LinearClassifier

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)

parser = argparse.ArgumentParser(description = 'Pytorch Image Training')
parser.add_argument('--lr', default=0.001, type = float, help = 'learning rate')
parser.add_argument('--resume', '-r', action = 'store_true', help = 'resume from checkpoint')
args = parser.parse_args()

if args.resume:
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

def train (model, optimizer, criterion, trainloader, epochs):
    '''Training the model using the variables: model, optimizer, criterion, trainloader, number of epochs'''
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), datat[1].to(device)

            optimizer.zero_grad() #zero the gradients of the model parameters 

            #forward + backward + optimize 
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            #performing the backpropogation step and updating the weights of the model parameters
            loss.backward()
            optimizer.step()

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

model = CNNClassifier() if args.model == "cnn" else LinearClassifier()

dataset = CIFAR10() if args.model == "cifar" else MNIST()

criterion = nn.CrossEntropyLoss()       #Loss function to calculate the difference between input and target
optimizer = optim.Adam(model.parameters(), lr = 0.001)      #Adam optimizer 
