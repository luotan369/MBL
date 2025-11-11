import ast
import argparse
import os
import json
import torch
import torch.backends.cudnn as cudnn
import torchvision
import numpy as np
import torchvision.transforms as transforms
from cifar10_models.resnet import ResNet18, ResNet34
from cifar10_models.vgg import VGG
from cifar10_models.pyramidnet import pyramid_net110
from cifar10_models.densenet import DenseNet121
from cifar10_models.preact_resnet import PreActResNet101


def prepare_normed_data():
    print('==> Preparing data..')
    means = [0.4914, 0.4822, 0.4465]
    stds = [0.2023, 0.1994, 0.2010]
    normalize = transforms.Normalize(mean=np.array(means), std=np.array(stds))
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize
    ])

    trainset = torchvision.datasets.CIFAR10(root=path, train=True, download=True, transform=transform_train)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=path, train=False, download=True, transform=transform_test)

    testloader = torch.utils.data.DataLoader(testset, batch_size=100, num_workers=2)
    return trainloader, testloader


def prepare_data():
    print('==> Preparing data..')


    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor()
    ])


    trainset = torchvision.datasets.CIFAR10(root=path, train=True, download=True, transform=transform_train)


    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=path, train=False, download=True, transform=transform_test)


    testloader = torch.utils.data.DataLoader(testset, batch_size=100, num_workers=2)

    return trainloader, testloader


def prepare_model(model_name):
    if model_name == 'Resnet18':
        model = ResNet18()
    elif model_name == 'Resnet34':
        model = ResNet34()
    elif model_name == 'VGG13':
        model = VGG('VGG13')
    elif model_name == 'VGG19':
        model = VGG('VGG19')
    elif model_name == 'PreactResnet101':
        model = PreActResNet101()
    elif model_name == 'Densenet121':
        model = DenseNet121()
    elif model_name == 'Pyramidnet110':
        model = pyramid_net110()
    else:
        assert False
    model = model.cuda()
    cudnn.benchmark = True
    criterion = torch.nn.CrossEntropyLoss() 
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-6, nesterov=True)
    torch.manual_seed(1)
    return model, criterion, optimizer


def train(net, criterion, epoch, optimizer, trainloader):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    print('Train Loss: %.3f | Train Acc: %.3f%% (%d/%d)'
          % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))


def test(net, criterion, testloader):
    global BEST_ACC
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        print('Test Loss: %.3f | Test Acc: %.3f%% (%d/%d)'
              % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    # Save checkpoint.
    acc = 100. * correct / total
    return acc


# def save(model_name, if_top_5, net, acc, epoch):
def save(model_name, net, acc, epoch):
    print('Saving..')
    state = {
        'net': net.state_dict(),
        'acc': acc,
        'epoch': epoch,
    }
    if not os.path.isdir('cifar10'):
        os.mkdir('cifar10')
    # save_path = './cifar10/cifar10_{}_{}_ckpt.t7'.format(model_name, 'top_5' if if_top_5 else 'last_5')
    save_path = '../checkpoints/cifar10_target_models/{}_ckpt.t7'.format(model_name)
    print('save path', save_path)
    if torch.__version__ == '1.3.0':
        torch.save(state, save_path)
    else:
        torch.save(state, save_path, _use_new_zipfile_serialization=False)


def save_normed(model_name, net, acc, epoch):
    print('Saving..')
    state = {
        'net': net.state_dict(),
        'acc': acc,
        'epoch': epoch,
    }
    if not os.path.isdir('cifar10'):
        os.mkdir('cifar10')
    save_path = '../checkpoints/cifar10_target_models/{}_normed_ckpt.t7'.format(model_name)
    print('save path', save_path)
    if torch.__version__ == '1.3.0':
        torch.save(state, save_path)
    else:
        torch.save(state, save_path, _use_new_zipfile_serialization=False)


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def main():
    model_name = args.model_name
    # if_top_5 = args.if_top_5
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    model, criterion, optimizer = prepare_model(model_name)
    # trainloader, testloader = prepare_data(args.if_top_5)
    if args.norm:
        trainloader, testloader = prepare_normed_data()
    else:
        trainloader, testloader = prepare_data()

    best_acc = 0
    for epoch in range(start_epoch, start_epoch + 200):
        if epoch > 50 and epoch < 150:
            adjust_learning_rate(optimizer, 0.01)
        elif epoch >= 150:
            adjust_learning_rate(optimizer, 0.001)
        train(model, criterion, epoch, optimizer, trainloader)
        acc = test(model, criterion, testloader)
        if acc > best_acc:
            best_acc = acc
            # save(model_name, args.if_top_5, model, acc, epoch)
            if args.norm:
                save_normed(model_name, model, acc, epoch)
            else:
                save(model_name, model, acc, epoch)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train CIFAR10 models.')
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--norm", action='store_true')  
    args = parser.parse_args()
    path = '../data'
    main()
