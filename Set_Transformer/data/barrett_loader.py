import os, sys

import random
import torch
import matplotlib.pyplot as plt

import numpy  as np
import PIL
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data
from torchvision import datasets
import json
from sklearn.preprocessing import StandardScaler
class BarrettData(data.Dataset):
    def __init__(self, img_dir, batch_size=64, validation=False):
        self.set_size = batch_size
        self.img_dir = img_dir
        self.X, self.Y = "X", "Y"
        self.training_data = os.listdir(os.path.join(self.img_dir, self.X))
            
        self.validation = validation
        print("Training Data length: ", len(self.training_data))

    def __getitem__(self, index):
        #print(index)
        wsi_idx = self.training_data[index]
        wsi_patch_tensor = torch.load(os.path.join(self.img_dir,self.X,wsi_idx))
        patches = wsi_patch_tensor.shape[0]
        wsi_patch_labels = torch.load(os.path.join(self.img_dir,self.Y,wsi_idx))[:,0]
        #print("num of patches: ", patches)
        if patches < 16:
            batch = torch.randint(0, patches,(self.set_size,))
            x = wsi_patch_tensor[batch, :].to("cpu")
            y = wsi_patch_labels[batch].to("cpu")

        else:
        
            if self.validation:
             
                patch_logits = torch.load(os.path.join("./data/WSI_patches/bolero_20x_biopsies_last_full",self.X,wsi_idx))
            else:
                patch_logits = torch.load(os.path.join("./data/WSI_patches/train_full_20x_last_output",self.X,wsi_idx))

            patch_probs = torch.softmax(patch_logits,dim=1)
            patch_args_sorted = torch.argsort(patch_probs, dim=0, descending=True)

            patch_args_sorted = patch_args_sorted[:16,:]

            wsi_patch_labels = torch.load(os.path.join(self.img_dir,self.Y,wsi_idx))[:,0]

            n_patches, _ = wsi_patch_tensor.shape
            x = torch.zeros(64,512)
            y = torch.zeros(64)
            # for loop this instead of hardcoded
            x[:16,:] = wsi_patch_tensor[patch_args_sorted[:, 0],:]
            x[16:32,:] = wsi_patch_tensor[patch_args_sorted[:,1],:]
            x[32:48,:] = wsi_patch_tensor[patch_args_sorted[:,2],:]
            x[48:,:] = wsi_patch_tensor[patch_args_sorted[:,3],:]
            y[:16] = wsi_patch_labels[patch_args_sorted[:,0]]
            y[16:32] = wsi_patch_labels[patch_args_sorted[:,1]]
            y[32:48] = wsi_patch_labels[patch_args_sorted[:,2]]
            y[48:] = wsi_patch_labels[patch_args_sorted[:,3]]

        y = torch.unique(y)
        y = torch.tensor([2]) if (2 in y) else torch.tensor([3]) if (3 in y) else torch.tensor([y[-1]])

        return x, y, wsi_idx   

    def __len__(self):
        return len(self.training_data)
    