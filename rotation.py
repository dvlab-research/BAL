'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.allow_tf32 = True

import os

import argparse
import random
import time

from utils import setup_logger
from custom_datasets import trans_dict
from models import models_dict
from loader import RotationLoader

def get_args():
    parser = argparse.ArgumentParser(description='Self-supervised training')
    # Data-related arguments
    parser.add_argument('--datapath', default='DATAPATH', type=str, help='Path to the dataset.')
    parser.add_argument('--dataset', '-d', default='cifar10', type=str, help='Name of the dataset.')

    # Model-related arguments
    parser.add_argument('--net', '-n', default='vgg16', type=str, help='Name of the neural network model.')

    # Training-related arguments
    parser.add_argument('--batch_size', '-b', default=256, type=int, help='Batch size for training.')
    parser.add_argument('--save', default='', type=str, help='Path to save the trained model.')
    parser.add_argument('--epochs', default=120, type=int, help='Number of training epochs.')
    parser.add_argument('--print_freq', default=100, type=int, help='Print frequency during training.')
    parser.add_argument('--start_epoch', default=0, type=int, help='Epoch to start training from.')
    parser.add_argument('--lr', default=0.1, type=float, help='Learning rate for training.')
    parser.add_argument("--milestone", nargs='+', type=int, default=[30, 60, 90], help='List of epoch milestones for learning rate schedule.')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum for optimizer.')
    parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for optimizer.')
    args = parser.parse_args()
    return args

def train(epoch):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, inputs1, inputs2, inputs3, targets, targets1, targets2, targets3) in enumerate(trainloader):
        
        inputs, inputs1, targets, targets1 = inputs.to(device), inputs1.to(device), targets.to(device), targets1.to(device)
        inputs2, inputs3, targets2, targets3 = inputs2.to(device), inputs3.to(device), targets2.to(device), targets3.to(device)
        optimizer.zero_grad()
        outputs, outputs1, outputs2, outputs3 = net(inputs), net(inputs1), net(inputs2), net(inputs3)

        loss1 = criterion(outputs, targets)
        loss2 = criterion(outputs1, targets1)
        loss3 = criterion(outputs2, targets2)
        loss4 = criterion(outputs3, targets3)
        loss = (loss1+loss2+loss3+loss4)/4.
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        _, predicted1 = outputs1.max(1)
        _, predicted2 = outputs2.max(1)
        _, predicted3 = outputs3.max(1)
        total += targets.size(0)*4

        correct += predicted.eq(targets).sum().item()
        correct += predicted1.eq(targets1).sum().item()
        correct += predicted2.eq(targets2).sum().item()
        correct += predicted3.eq(targets3).sum().item()

        if batch_idx % args.print_freq == 0:
            print('Train:[{}][{}/{}] Loss: {:.3f} | Acc: {:.3f}'
            .format(epoch, batch_idx, len(trainloader), train_loss/(batch_idx+1), 100.*correct/total))

    logger.info('Train:[{}] Loss: {:.3f} | Acc: {:.3f}'.format(epoch, train_loss/(batch_idx+1), 100.*correct/total))
    
def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, inputs1, inputs2, inputs3, targets, targets1, targets2, targets3, path) in enumerate(testloader):
            inputs, inputs1, targets, targets1 = inputs.to(device), inputs1.to(device), targets.to(device), targets1.to(device)
            inputs2, inputs3, targets2, targets3 = inputs2.to(device), inputs3.to(device), targets2.to(device), targets3.to(device)
            outputs = net(inputs)
            outputs1 = net(inputs1)
            outputs2 = net(inputs2)
            outputs3 = net(inputs3)
            loss1 = criterion(outputs, targets)
            loss2 = criterion(outputs1, targets1)
            loss3 = criterion(outputs2, targets2)
            loss4 = criterion(outputs3, targets3)
            loss = (loss1+loss2+loss3+loss4)/4.
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            _, predicted1 = outputs1.max(1)
            _, predicted2 = outputs2.max(1)
            _, predicted3 = outputs3.max(1)
            total += targets.size(0)*4

            correct += predicted.eq(targets).sum().item()
            correct += predicted1.eq(targets1).sum().item()
            correct += predicted2.eq(targets2).sum().item()
            correct += predicted3.eq(targets3).sum().item()

            if batch_idx % args.print_freq == 0:
                print('Test:[{}][{}/{}] Loss: {:.3f} | Acc: {:.3f}'
                .format(epoch, batch_idx, len(testloader), test_loss/(batch_idx+1), 100.*correct/total))
        logger.info('Test:[{}] Loss: {:.3f} | Acc: {:.3f}'.format(epoch, test_loss/(batch_idx+1), 100.*correct/total))

    # Save checkpoint.
    acc = 100.*correct/total
    with open(os.path.join(args.save, 'best_rotation.txt'),'a') as f:
        f.write(str(acc)+':'+str(epoch)+'\n')
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        # save rotation weights
        torch.save(state, os.path.join(args.save, 'rotation.pth'))
        best_acc = acc

def get_datasets():
    print('==> Loading dataset {}..'.format(args.dataset))
    transform_train, transform_test = trans_dict[args.dataset]
    trainset = RotationLoader(path=os.path.join(args.datapath, args.dataset), is_train=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    testset = RotationLoader(path=os.path.join(args.datapath, args.dataset), is_train=False,  transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    return trainloader, testloader

if __name__ == '__main__':
    args = get_args()

    device = 'cuda'
    best_acc = 0  # best test accuracy

    trainloader, testloader = get_datasets()

    net = models_dict[args.net](num_classes=4)
    net = net.to(device)
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestone)

    if not os.path.isdir(args.save):
        os.makedirs(args.save)
    logger = setup_logger(name='Rotation', output=args.save)
    logger.info(args)

    print('==> Training..')
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        train(epoch)
        test(epoch)
        scheduler.step()
        present_time = time.time()
        eta = (present_time - start_time)/(epoch - args.start_epoch + 1) *(args.epochs - epoch - 1)
        eta = time.strftime("%dd %H:%M:%S", time.gmtime(eta))
        print('Eta: {}'.format(eta))