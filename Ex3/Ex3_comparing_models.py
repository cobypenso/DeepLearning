# Coby Penso, 208254128
import numpy as np
from numpy.core.fromnumeric import size
import pandas as pd
import matplotlib.pyplot as plt
from utils import *
import argparse
import torch
from torchvision import transforms
import torchvision
from models import *

use_cuda = torch.cuda.is_available()
seed = 42
# Set seed
np.random.seed(seed)
torch.manual_seed(seed)
if use_cuda:
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
# Handel GPU stochasticity
torch.backends.cudnn.enabled = use_cuda
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if use_cuda else "cpu")

def train(net, trainloader, validloader, optimizer, criterion ,numEpochs):
    train_loss = 0

    net = net.to(device)

    for epoch in range(numEpochs):  # loop over the dataset multiple times
        net.train()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            train_loss = running_loss
            if i % 10 == 0:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0
            # test on validation dataset
        valid_acc = test(net, validloader)

    print('Finished Training')
    return train_loss, valid_acc

def test(net, testloader):
    correct = 0
    total = 0
    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    net.train()
    return (100 * correct / total)

def get_loaders(train_transform, test_transform, batch_size):
    trainset = torchvision.datasets.STL10(root='./data', split='train',
                                            download=True, transform=train_transform)
    trainset, validset = split_train_and_val(trainset, 0.8)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=1)
    validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size,
                                            shuffle=True, num_workers=1)
    testset = torchvision.datasets.STL10(root='./data', split='test',
                                        download=True, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=1)
    
    return trainloader, validloader, testloader

def get_optimizer(model_params, optimizer_type, lr, weight_decay):

    if optimizer_type == 'SGD':
        optimizer = optim.SGD(model_params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_type == 'Adam':
        optimizer = optim.Adam(model_params, lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)
    elif optimizer_type == 'RMSProp':
        optimizer = optim.Adam(model_params, lr=lr, alpha=0.99, eps=1e-08, weight_decay=weight_decay)
    else:
        NotImplementedError("optimizer not implemented")
    return optimizer

def hyperparam_search(net, net_params_to_learn, criterion, train_transform, test_transform):
    
    numEpochs = 60

    results = []
    for lr in [0.01, 0.001]:
        for bt in [16, 64, 128]:
            for wd in [0.01, 0.001]:
                for opt_type in ['SGD', 'Adam', 'RMSProp']:
                    optimizer = get_optimizer(net_params_to_learn, opt_type, lr, weight_decay=wd)
                    trainloader, validloader, testloader = get_loaders(train_transform, test_transform, bt)
                    
                    train_loss, valid_accuracy = train(net, trainloader, validloader, optimizer, criterion, numEpochs)
                    test_accuracy = test(net, testloader)

                    results.append({'lr': lr,
                                    'wd': wd,
                                    'bt': bt,
                                    'optimizer_type': opt_type,
                                    'loss': train_loss,
                                    'test_accuracy': test_accuracy})


###########################################
################### Main ##################
###########################################

def main(args):
    mean, std = (0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)

    train_transform = transforms.Compose(
        [transforms.Resize(48),
        transforms.RandomCrop(32),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)]) 

    test_transform = transforms.Compose(
        [transforms.Resize(48),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)])
    
    if args.visualize:
        trainset = torchvision.datasets.STL10(root='./data', split='train',
                                            download=True, transform=train_transform)
        visualize(trainset, 4)

    numEpochs = 20
    hidden_sizes = [1000, 500, 100, 10]
    num_classes = 10
    image_dim = 32 * 32 * 3
    criterion = nn.CrossEntropyLoss()
    
    if args.net == 'LR':
        #Logistic Regression net
        logistic_regression_net = LogisticRegression(image_dim, num_classes)
        hyperparam_search(logistic_regression_net,
                          logistic_regression_net.parameters(), 
                          criterion,
                          train_transform, test_transform)
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
    parser.add_argument("--net", default="LR", type=str, choices=['LR', 'FC3', 'CNN','ResNet18_fine_tune', 'ResNet18_feature_extractor'])
    parser.add_argument("--optimizer_type", default="SGD", type=str, choices=['SGD', 'Adam', 'RMSProp'])
    parser.add_argument("--visualize", default=False, type=bool)
    args = parser.parse_args()
    main(args)