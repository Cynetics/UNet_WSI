import torchvision.transforms as transforms
from PIL import Image
from .barrett_loader import *
import pathlib

def get_barrett_data(config=None):
    print(pathlib.Path().absolute())
    #BarrettData(img_dir="../../../../prostate-cancer-grade-assessment/train_patches/",
    #                            mask_dir="../../../../prostate-cancer-grade-assessment/train_patches_masks/",
    #                            image_transform=True
    #                           )
    
    #img_dir="../Patches_10x/train/",
    #mask_dir="../Patches_10x/three_five_seven/train_masks"
    

    training_data = BarrettData(img_dir="../Patches_20x/train/",
                                mask_dir="../Patches_20x/three_five_seven/train_masks",
                                image_transform=True, train=False
                               )
    return training_data

def get_barrett_data_regression(config=None):
    print(pathlib.Path().absolute())
    #BarrettData(img_dir="../../../../prostate-cancer-grade-assessment/train_patches/",
    #                            mask_dir="../../../../prostate-cancer-grade-assessment/train_patches_masks/",
    #                            image_transform=True
    #                           )
    
    #img_dir="../Patches_10x/train/",
    #mask_dir="../Patches_10x/three_five_seven/train_masks"
    

    training_data = BarrettDataRegression(img_dir="../Patches_20x/train/",
                                mask_dir="../Patches_20x/three_five_seven/train_masks",
                                image_transform=True, train=False
                               )
    return training_data


def get_barrett_val_data_regression(config=None):
    val_data = BarrettDataRegression(img_dir="../Bolero/test_20x/" ,
                                mask_dir="../Bolero/three_five_seven/test_20x_masks/",
                                image_transform=False, train=False
                               )
    return val_data


def get_barrett_val_data(config=None):
    val_data = BarrettData(img_dir="../Bolero/test_20x/" ,
                                mask_dir="../Bolero/three_five_seven/test_20x_masks/",
                                image_transform=False, train=False
                               )
    return val_data
