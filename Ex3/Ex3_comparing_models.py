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
import pandas as pd  

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
            # test on validation dataset
        train_loss = running_loss / i
        valid_acc = test(net, validloader)
        print ("epoch: " + str(epoch) + " valid accuracy: " + str(valid_acc) + " train loss: " + str(train_loss))

    print('Finished Training')
    return train_loss, valid_acc

def test(net, testloader):
    net.eval()
    correct = 0
    total = 0
    net.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            
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
                                            shuffle=True, num_workers=0)
    validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size,
                                            shuffle=True, num_workers=0)
    testset = torchvision.datasets.STL10(root='./data', split='test',
                                        download=True, transform=test_transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=0)
    
    return trainloader, validloader, testloader

def get_optimizer(model, optimizer_type, lr, weight_decay):

    params_to_update = []
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
    if optimizer_type == 'SGD':
        optimizer = optim.SGD(params_to_update, lr=lr, momentum=0.9)
    elif optimizer_type == 'Adam':
        optimizer = optim.Adam(params_to_update, lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)
    elif optimizer_type == 'RMSProp':
        optimizer = optim.RMSProp(params_to_update, lr=lr, alpha=0.99, eps=1e-08, weight_decay=weight_decay)
    else:
        NotImplementedError("optimizer not implemented")
    return optimizer

def hyperparam_search(net, criterion, train_transform, test_transform):
    
    numEpochs = 200

    results = []
    for lr in [0.01, 0.001]:
        for bt in [32, 64, 128]:
            for wd in [0, 0.01]:
                for opt_type in ['SGD', 'Adam']:
                    net.apply(init_weights)
                    optimizer = get_optimizer(net, opt_type, lr, weight_decay=wd)
                    trainloader, validloader, testloader = get_loaders(train_transform, test_transform, bt)
                    
                    train_loss, valid_accuracy = train(net, trainloader, validloader, optimizer, criterion, numEpochs)
                    test_accuracy = test(net, testloader)
                    print(test_accuracy)
                    results.append({'lr': lr,
                                    'wd': wd,
                                    'bt': bt,
                                    'optimizer_type': opt_type,
                                    'loss': train_loss,
                                    'test_accuracy': test_accuracy})
    return results


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

    hidden_sizes = [1000, 500, 100, 10]
    num_classes = 10
    image_dim = 32 * 32 * 3
    criterion = nn.CrossEntropyLoss()
    
    if args.net == 'LR':
        #Logistic Regression net
        logistic_regression_net = LogisticRegression(image_dim, num_classes, device)
        results = hyperparam_search(logistic_regression_net,
                          criterion,
                          train_transform, test_transform)
                          
    elif args.net == 'FC3':
        #FullyConnected 3Hidden Layers+Dropout+BN
        fc_net = FC3_Net(image_dim, num_classes, device)
        results = hyperparam_search(fc_net,
                          criterion,
                          train_transform, test_transform)
        
    elif args.net == 'CNN':
        #CNN with 2 Conv layers
        cnn_net = CNN_Net(image_dim, num_classes, device)
        results = hyperparam_search(cnn_net,
                          criterion,
                          train_transform, test_transform)
                          
    elif args.net =='ResNet18_fine_tune':
        #ResNet18 Fine Tuning of all the paramters of the net
        resnet18_fine_tuning = PreTrained_ResNet18(hidden_sizes, num_classes, False, device)
        resnet18_fine_tuning.apply(init_weights)
        results = hyperparam_search(resnet18_fine_tuning,
                          criterion,
                          train_transform, test_transform)

    elif args.net == 'ResNet18_feature_extractor':
        #ResNet18 as only a feature extractor
        resnet18_feature_extractor_only = PreTrained_ResNet18(hidden_sizes, num_classes, True, device)
        resnet18_feature_extractor_only.apply(init_weights)
        results = hyperparam_search(resnet18_feature_extractor_only,
                          criterion,
                          train_transform, test_transform)
    else:
        raise NotImplementedError

    df = pd.DataFrame(results)
    df.to_csv(args.net + '60_epochs_summary.csv')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Network")
    parser.add_argument("--net", default="ResNet18_feature_extractor", type=str, choices=['LR', 'FC3', 'CNN','ResNet18_fine_tune', 'ResNet18_feature_extractor'])
    parser.add_argument("--optimizer_type", default="SGD", type=str, choices=['SGD', 'Adam', 'RMSProp'])
    parser.add_argument("--visualize", default=False, type=bool)
    args = parser.parse_args()
    main(args)