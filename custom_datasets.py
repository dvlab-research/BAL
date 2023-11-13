import os
import random
import numpy as np
import torch
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torchvision
from skimage import transform

def caltech_transformer():
    train_transform = transforms.Compose([
        transforms.Lambda(lambda x: x.convert("RGB") ),
        transforms.RandomHorizontalFlip(),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                            std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404]),
    ])
    test_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.Lambda(lambda x: x.convert("RGB") ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404]),
        ])
    return train_transform, test_transform

def svhn_transformer():
    train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404]),
        ])

    test_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5070751592371323, 0.48654887331495095, 0.4409178433670343],
                                std=[0.2673342858792401, 0.2564384629170883, 0.27615047132568404]),
        ])
    return train_transform, test_transform

def cifar_transformer():
    train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465,],
                                std=[0.247, 0.243, 0.261,]),
        ])
    test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465,],
                                std=[0.247, 0.243, 0.261,]),
        ])

    return train_transform, test_transform

def imagenet_transformer():
    train_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    test_transform = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return train_transform, test_transform

cls_dict = {
    'cifar10': 10,
    'cifar100': 100, 
    'caltech101': 101, 
    'caltech256': 256,
    'svhn': 10,
    'imagenet100': 100,
    'tinyimagenet': 200,
}

trans_dict = {
    'cifar10': cifar_transformer(),
    'cifar100': cifar_transformer(), 
    'caltech101': caltech_transformer(), 
    'caltech256': caltech_transformer(),
    'imagenet100': imagenet_transformer(),
    'tinyimagenet': imagenet_transformer(),
    'svhn': svhn_transformer(),
}

def get_num_images(dataset):
    if dataset == 'cifar10':
        num_images = 50000
        num_val = 5000
        num_classes = 10
    elif dataset == 'imagenet100':
        num_images = 130000
        num_classes = 100
    elif dataset == 'tinyimagenet':
        num_images = 100000
        num_classes = 200
    elif dataset == 'cifar100':
        num_val = 5000
        num_images = 50000
        num_classes = 100
    elif dataset == 'caltech256':
        num_val = 3000
        num_images = 27607
        num_classes = 256
    elif dataset == 'caltech101':
        num_val = 914
        num_images = 7857
        num_classes = 101
    elif dataset == 'svhn':
        num_images = 73257
        num_val = 7325
        num_classes = 10
    else:
        raise NotImplementedError   

    return num_images