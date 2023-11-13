import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from loader import RotationLoader
import os, glob
import argparse
from custom_datasets import trans_dict, cls_dict
from utils import setup_logger
from tqdm import tqdm
from models import models_dict

def get_args():
    parser = argparse.ArgumentParser(description='Kmeans Clustering')
    parser.add_argument('--dataset', '-d', default='cifar10', type=str, help='Name of the dataset.')
    parser.add_argument('--net', '-n', default='vgg16', type=str, help='Name of the neural network model.')
    parser.add_argument('--datapath', default='DATAPATH', type=str, help='Path to the dataset.')
    parser.add_argument('--kmeans', default='Kmeans', choices=['MiniBatch', 'Kmeans'], type=str, help='Choose between MiniBatch and Kmeans.')
    parser.add_argument('--load', required=True, default='LOADPATH', type=str, help='Path dir of the rotation.pth to load.')
    args = parser.parse_args()
    return args

def save(name, file):
    np.save(name, file)
    print(name + ' saved!')

def load(name, allow_pickle=True):
    file = np.load(name, allow_pickle=allow_pickle) 
    print(name + ' loaded!')
    return file

def test(epoch):
    paths = []
    _, transform_test = trans_dict[args.dataset]

    print(f'=> Loading dataset {args.dataset}')
    testset = RotationLoader(path=os.path.join(args.datapath, args.dataset), is_train=False,  transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=2)

    print(f'=> Loading net {args.net}')
    net = models_dict[args.net](num_classes=4)

    net = net.to(device)
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

    checkpoint = torch.load(os.path.join(args.load, 'rotation.pth'))
    net.load_state_dict(checkpoint['net'])

    feats = []
    net.eval()
    losses = []
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        with tqdm(total=len(testloader), desc=f"Extracting features") as pbar:
            for idx, (inputs, inputs1, inputs2, inputs3, targets, targets1, targets2, targets3, path) in enumerate(testloader):
                inputs, inputs1, targets, targets1 = inputs.to(device), inputs1.to(device), targets.to(device), targets1.to(device)
                inputs2, inputs3, targets2, targets3 = inputs2.to(device), inputs3.to(device), targets2.to(device), targets3.to(device)
                outputs, feat = net(inputs, is_feat=True)
                outputs1 = net(inputs1)
                outputs2 = net(inputs2)
                outputs3 = net(inputs3)
                loss1 = criterion(outputs, targets)
                loss2 = criterion(outputs1, targets1)
                loss3 = criterion(outputs2, targets2)
                loss4 = criterion(outputs3, targets3)
                loss = (loss1 + loss2 + loss3 + loss4)/4.
                losses.append(loss.item())
                feats.append(np.array(torch.squeeze(feat).cpu()))
                paths.append(path)
                pbar.update(1)

    save(feats_name, feats)
    save(paths_name, paths)
    save(losses_name, losses)

def kmeans_train(feats, batch_size):
    # 3, building the kmeans model
    print('=> Loading kmeans model')
    if args.kmeans == 'MiniBatch':
        kMeansModel = MiniBatchKMeans(init='k-means++', n_clusters=num_classes, batch_size=batch_size)
        for i in range(0, len(feats), batch_size):
            batch = feats[i: i + batch_size]
            kMeansModel = kMeansModel.partial_fit(batch)
    elif args.kmeans == 'Kmeans':
        kMeansModel = KMeans(init='k-means++', n_clusters=num_classes)  
        kMeansModel.fit(feats)
    else:
        raise

    distances = kMeansModel.transform(feats)

    save(distances_name, distances)
    print(distances_name + ' saved!')

def save_unlabeled_pool(name, data):
    if os.path.exists(name):
        os.system(f'rm {name}')
    with tqdm(total=len(data), desc=f"Saving unlabeled pool") as pbar:
        with open(name, 'a') as f:
            for item in data:
                f.write(f'{item[0]}\n')
                pbar.update(1)

if __name__ == "__main__":
    args = get_args()
    device = 'cuda:0'
    num_classes = cls_dict[args.dataset]

    if not os.path.isdir(args.load):
        os.makedirs(args.load)
    logger = setup_logger(name='kmeans', output=args.load)
    logger.info(args)

    distances_name = os.path.join(args.load, 'feats_distances.npy')
    feats_name = os.path.join(args.load, "feats_feats.npy")
    paths_name = os.path.join(args.load, "feats_paths.npy")
    losses_name = os.path.join(args.load, "losses.npy")
    save_name = os.path.join(args.load, "unlabeled_pool.txt")
    os.makedirs(args.load, exist_ok=True)

    if not os.path.exists(distances_name):
        if not os.path.exists(feats_name) or not os.path.exists(paths_name):
            test(1)
        feats = list(load(feats_name))

    paths = load(paths_name) 
    if not os.path.exists(distances_name):
        kmeans_train(feats, batch_size=10*num_classes)
    distances = load(distances_name) 
    dds = np.sort(distances, axis=1)[:, 1] - np.sort(distances, axis=1)[:, 0]
    sort_index = np.argsort(dds)
    output = paths[sort_index]
    save_unlabeled_pool(save_name, output)
    print('done')










