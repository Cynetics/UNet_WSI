import os, sys

import random
import torch
import matplotlib.pyplot as plt

import numpy  as np
import PIL
from PIL import Image
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data
from torchvision import datasets
import json

class BarrettData(data.Dataset):
    def __init__(self, img_dir, mask_dir, train=True, image_transform=False):
        self.datalist = img_dir
        self.mask_dir = mask_dir
        self.data = os.listdir(self.datalist)
        # training size is number of WSIs in training set
        print("training set length: ", len(self.data))
        self.image_transform = image_transform
        self.train = train

    def __getitem__(self, index):
        img_path = os.path.join(self.datalist, self.data[index])
        mask_path = os.path.join(self.mask_dir, self.data[index])
        img = Image.open(img_path)
        mask = Image.open(mask_path)
        
        if self.train:
            mask_accept = False

            while not mask_accept:
                np_mask = np.array(mask)
                uniques = list(np.unique(np_mask))
                if not 2 in uniques and not 3 in uniques:
                    rand = np.random.rand()
                    if rand > 0.80:
                        mask_accept = True
                    else:
                        # 36380 num of datapoints in 10x data
                        rand_idx = np.random.randint(0,36380)
                        img_path = os.path.join(self.datalist, self.data[rand_idx])
                        mask_path = os.path.join(self.mask_dir, self.data[rand_idx])
                        img = Image.open(img_path)
                        mask = Image.open(mask_path)
                else:
                    mask_accept = True

        
        if self.image_transform:
            img, mask = self.transform(img, mask)
        
        #img = (TF.to_tensor(img) - 0.5) / 0.5
        # ImageNet Encoder Normalization
        #img = (TF.to_tensor(img) - 0.5) / 0.5
        img = TF.to_tensor(img)
        #img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        mask = torch.tensor(np.array(mask))
        #print(torch.unique(mask))
        #print(img)
        return img, mask

    def transform(self, img, mask):
        """
        Data augmentation pipeline, we could experiment
        with colour and the distortions.
        At the moment, we are not performing colour
        augmentation.
        """
        i, j, h, w = transforms.RandomCrop.get_params(img, output_size=(384,384))
        #print(i,j,h,w)
        i = np.random.randint(256,512)
        j = np.random.randint(0,256)
        if random.random() > 0.1:
            img = TF.resized_crop(img, i,j,h,w, size=512)
            mask = TF.resized_crop(mask, i,j,h,w, size=512)
        if random.random() > 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)
        if random.random() > 0.5:
            img = TF.vflip(img)
            mask = TF.vflip(mask)
        if random.random() > 0.5:
            img = TF.rotate(img, 90)
            mask = TF.rotate(mask, 90)        
        return img, mask

    def __len__(self):
        return len(self.data)
    
    
class BarrettDataRegression(data.Dataset):
    def __init__(self, img_dir, mask_dir, train=True, image_transform=False):
        self.datalist = img_dir
        self.mask_dir = mask_dir
        self.data = os.listdir(self.datalist)
        # training size is number of WSIs in training set
        print("training set length: ", len(self.data))
        self.image_transform = image_transform
        self.train = train

    def __getitem__(self, index):
        img_path = os.path.join(self.datalist, self.data[index])
        mask_path = os.path.join(self.mask_dir, self.data[index])
        img = Image.open(img_path)
        mask = Image.open(mask_path)
        
        if self.train:
            mask_accept = False

            while not mask_accept:
                np_mask = np.array(mask)
                uniques = list(np.unique(np_mask))
                if not 2 in uniques and not 3 in uniques:
                    rand = np.random.rand()
                    if rand > 0.80:
                        mask_accept = True
                    else:
                        # 36380 num of datapoints in 10x data
                        rand_idx = np.random.randint(0,36380)
                        img_path = os.path.join(self.datalist, self.data[rand_idx])
                        mask_path = os.path.join(self.mask_dir, self.data[rand_idx])
                        img = Image.open(img_path)
                        mask = Image.open(mask_path)
                else:
                    mask_accept = True

        
        if self.image_transform:
            img, mask = self.transform(img, mask)
        

        img = TF.to_tensor(img)
        #img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        mask = torch.tensor(np.array(mask))
        
        mask_out = torch.zeros(mask.shape)
        mask_out[torch.where(mask==3)] = 2
        mask_out[torch.where(mask==2)] = 3
        mask_out[torch.where(mask==1)] = 1
        #print(mask_out)
        mask_out = mask_out / 3
        #print(torch.unique(mask_out))
        #raise ValueError()
        #print(img)
        return img, mask

    def transform(self, img, mask):
        """
        Data augmentation pipeline, we could experiment
        with colour and the distortions.
        At the moment, we are not performing colour
        augmentation.
        """
        i, j, h, w = transforms.RandomCrop.get_params(img, output_size=(384,384))
        #print(i,j,h,w)
        i = np.random.randint(256,512)
        j = np.random.randint(0,256)
        if random.random() > 0.1:
            img = TF.resized_crop(img, i,j,h,w, size=512)
            mask = TF.resized_crop(mask, i,j,h,w, size=512)
        if random.random() > 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)
        if random.random() > 0.5:
            img = TF.vflip(img)
            mask = TF.vflip(mask)
        if random.random() > 0.5:
            img = TF.rotate(img, 90)
            mask = TF.rotate(mask, 90)        
        return img, mask

    def __len__(self):
        return len(self.data)

