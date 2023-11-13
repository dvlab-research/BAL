import glob
import os
from PIL import Image, ImageFilter

from torch.utils.data import Dataset, DataLoader
import torch
import torchvision.transforms as transforms
import numpy as np
import random
import cv2

random.seed(0)

class RotationLoader(Dataset):
    def __init__(self, path, is_train=True, transform=None):
        self.is_train = is_train
        self.transform = transform
        # self.h_flip = transforms.RandomHorizontalFlip(p=1)
        if self.is_train == 0: # train
            self.img_path = glob.glob(os.path.join(path, 'train/*/*'))
        else:
            self.img_path = glob.glob(os.path.join(path, 'train/*/*'))

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_path[idx])
        img = Image.fromarray(img)
        if img == None:
            print(self.img_path[idx])

        if self.is_train:
            img = self.transform(img)
            img1 = torch.rot90(img, 1, [1,2])
            img2 = torch.rot90(img, 2, [1,2])
            img3 = torch.rot90(img, 3, [1,2])
            imgs = [img, img1, img2, img3]
            rotations = [0,1,2,3]
            random.shuffle(rotations)
            return imgs[rotations[0]], imgs[rotations[1]], imgs[rotations[2]], imgs[rotations[3]], rotations[0], rotations[1], rotations[2], rotations[3]
        else:
            img = self.transform(img)
            img1 = torch.rot90(img, 1, [1,2])
            img2 = torch.rot90(img, 2, [1,2])
            img3 = torch.rot90(img, 3, [1,2])
            imgs = [img, img1, img2, img3]
            rotations = [0,1,2,3]
            random.shuffle(rotations)
            return imgs[rotations[0]], imgs[rotations[1]], imgs[rotations[2]], imgs[rotations[3]], rotations[0], rotations[1], rotations[2], rotations[3], self.img_path[idx]

class RotationLoader2(Dataset):
    def __init__(self, path, is_train=True, transform=None):
        self.is_train = is_train
        self.transform = transform
        # self.h_flip = transforms.RandomHorizontalFlip(p=1)
        if self.is_train == 0: # train
            self.img_path = glob.glob(os.path.join(path, 'train/*/*'))
        else:
            self.img_path = glob.glob(os.path.join(path, 'train/*/*'))

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_path[idx])
        img = Image.fromarray(img)
        if img == None:
            print(self.img_path[idx])

        if self.is_train:
            img = self.transform(img)
            img1 = torch.rot90(img, 1, [1,2])
            img2 = torch.rot90(img, 2, [1,2])
            img3 = torch.rot90(img, 3, [1,2])
            imgs = [img, img1, img2, img3]
            rotations = [0,1,2,3]
            random.shuffle(rotations)
            return imgs[rotations[0]], imgs[rotations[1]], imgs[rotations[2]], imgs[rotations[3]], idx, idx, idx, idx
        else:
            img = self.transform(img)
            img1 = torch.rot90(img, 1, [1,2])
            img2 = torch.rot90(img, 2, [1,2])
            img3 = torch.rot90(img, 3, [1,2])
            imgs = [img, img1, img2, img3]
            rotations = [0,1,2,3]
            random.shuffle(rotations)
            return imgs[rotations[0]], imgs[rotations[1]], imgs[rotations[2]], imgs[rotations[3]], idx, idx, idx, idx, self.img_path[idx]

class Loader2(Dataset):
    def __init__(self, path, is_train=True, transform=None, path_list=None, noise_level=0):
        self.is_train = is_train
        self.transform = transform
        self.path_list = path_list
        self.num_classes = len(glob.glob(os.path.join(path, 'train/*')))
        if self.is_train: # train
            if path_list:
                self.img_path = [p[0] if type(p) is np.ndarray else p for p in path_list]
            else:
                self.img_path = glob.glob(os.path.join(path, 'train/*/*'))
        else:
            self.img_path = [p[0] if type(p) is np.ndarray else p for p in path_list]

        self.target_path = []
        for img in self.img_path:
            r = random.uniform(0,1)
            if(r < noise_level):
                target = int(random.uniform(0, self.num_classes))
            else:
                target = int(img.split('/')[-2])
            self.target_path.append(target)

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img = Image.open(self.img_path[idx]).convert('RGB')
        img = self.transform(img)
        target = self.target_path[idx]

        return img, target

class Loader(Dataset):
    def __init__(self, path, is_train=True, transform=None):
        self.is_train = is_train
        self.transform = transform
        if self.is_train: # train
            self.img_path = glob.glob(os.path.join(path, 'train/*/*'))
        else:
            self.img_path = glob.glob(os.path.join(path, 'test/*/*'))

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_path[idx])
        img = Image.fromarray(img).convert('RGB')
        img = self.transform(img)
        label = int(self.img_path[idx].split('/')[-2])

        return img, label
