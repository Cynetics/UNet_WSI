import torchvision.transforms as transforms
from PIL import Image
from .barrett_loader import *
import pathlib

def get_transformer_data(config=None):
    print(pathlib.Path().absolute())

    training_data = BarrettData(img_dir="./data/WSI_patches/train_full_20x/", batch_size=config.set_size, validation=False)
    return training_data

def get_transformer_val_data(config=None):
    print(pathlib.Path().absolute())
    val_data = BarrettData(img_dir="./data/WSI_patches/bolero_20x_biopsies_full/", batch_size=config.set_size, validation=True)
    return val_data