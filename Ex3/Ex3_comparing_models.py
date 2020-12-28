# Coby Penso, 208254128
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import *
import argparse
import torch
from torchvision import transforms
import torchvision
from models import *

def train(net, trainloader, optimizer, criterion ,numEpochs):
    for epoch in range(numEpochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 10 == 0:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0

    print('Finished Training')

def test(net, testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (
        100 * correct / total))

###########################################
################### Main ##################
###########################################

def main(args):
    mean, std = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize(mean, std)])

    trainset = torchvision.datasets.STL10(root='./data', split='train',
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                            shuffle=True, num_workers=0)

    testset = torchvision.datasets.STL10(root='./data', split='test',
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                            shuffle=False, num_workers=0)

    # classes = ('plane', 'car', 'bird', 'cat',
    #         'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    numEpochs = 20
    hidden_sizes = [1000, 500, 100, 10]
    num_classes = 10
    image_dim = 96 * 96 * 3
    criterion = nn.CrossEntropyLoss()
    
    if args.net == 'LR':
        #Logistic Regression net
        logistic_regression_net = LogisticRegression(image_dim, num_classes)
        optimizer = optim.SGD(logistic_regression_net.parameters(), lr=0.001, momentum=0.9)
        train(logistic_regression_net, trainloader, optimizer, criterion, numEpochs)
        test(logistic_regression_net, testloader)
    elif args.net == 'FC3':
        #FullyConnected 3Hidden Layers+Dropout+BN
        fc_net = FC3_Net(image_dim, num_classes)
        optimizer = optim.SGD(fc_net.parameters(), lr=0.001, momentum=0.9)
        train(fc_net, trainloader, optimizer, criterion, numEpochs)
        test(fc_net, testloader)
    elif args.net == 'CNN':
        #CNN with 2 Conv layers
        cnn_net = CNN_Net(image_dim, num_classes)
        optimizer = optim.SGD(cnn_net.parameters(), lr=0.001, momentum=0.9)
        train(cnn_net, trainloader, optimizer, criterion, numEpochs)
        test(cnn_net, testloader)
    elif args.net =='ResNet18_fine_tune':
        #ResNet18 Fine Tuning of all the paramters of the net
        resnet18_fine_tuning = PreTrained_ResNet18(hidden_sizes, num_classes)
        resnet18_fine_tuning.apply(init_weights)
        optimizer = optim.SGD(resnet18_fine_tuning.parameters(), lr=0.001, momentum=0.9)
        train(resnet18_fine_tuning, trainloader, optimizer, criterion, numEpochs)
        test(resnet18_fine_tuning, testloader)
    elif args.net == 'ResNet18_feature_extractor':
        #ResNet18 as only a feature extractor
        resnet18_feature_extractor_only = PreTrained_ResNet18(hidden_sizes, num_classes)
        resnet18_feature_extractor_only.apply(init_weights)
        optimizer = optim.SGD(resnet18_feature_extractor_only.net.parameters(), lr=0.001, momentum=0.9)
        train(resnet18_feature_extractor_only, trainloader, optimizer, criterion, numEpochs)
        test(resnet18_feature_extractor_only, testloader)
    else:
        raise NotImplementedError

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Network")
    parser.add_argument("--net", default="ResNet18_feature_extractor", type=str, choices=['LR', 'FC3', 'CNN','ResNet18_fine_tune', 'ResNet18_feature_extractor'])
    args = parser.parse_args()
    main(args)