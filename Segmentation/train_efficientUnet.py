from optparse import OptionParser
import numpy as np
# Torch dependencies
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim
# Our own written dependencies
from metrics import Metric
#from efficientunet import *
import segmentation_models_pytorch as smp
from torch.utils.data import DataLoader
from utils.config import dense_unet_config

from data.load_data import *
from scheduler import CycleScheduler

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import matplotlib.pyplot as plt

def train_net(net, training_loader, val_loader, config, save_cp=True, model_name="model"):
    print("Training: ", model_name)

    # The direcotry where we would like to save our trained models
    dir_checkpoint = '../Models/'

    # The optimizer, optim.SGD with momentum works just fine but Adam converges much faster
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=config.lr)

    # The weights are based on the proportion of each class
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1,1,1.5, 1.5]).float().cuda())
    classification_criterion = nn.CrossEntropyLoss(weight=torch.tensor([1,1,1.5,1.5]).float().cuda())
    metric = Metric(device=device)
    total_l = []
    class_l = []
    seg_l = []
    for i in range(config.iterations):
        imgs, true_masks = next(iter(training_loader))
        imgs = imgs.to(device)
        true_masks = true_masks.to(device).long().squeeze(1)
        #print(torch.unique(true_masks[0]))
        B = true_masks.shape[0]
        patch_labels = torch.zeros(B).to(device)
        for label in [0,1,3,2]:
            hard_labels = (true_masks.view(B, -1)==label).long()
            hard_labels = hard_labels.sum(dim=1)
            patch_labels[torch.where(hard_labels > (512*512*0.2))] = label
        patch_labels = patch_labels.long()

        masks_probs, classification = net(imgs)
        loss = criterion(masks_probs, true_masks)
        seg_l.append(loss.item())

        class_loss = classification_criterion(classification, patch_labels)
        class_l.append(class_loss.item())
        loss = loss + class_loss
        total_l.append(loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        #print('{} --- loss: {}'.format(i, np.mean(total_l[-100:])))
        if (i+1) % config.n_eval==0:
            print("[==========================]")
            print("Current Iteration: {}".format(i+1))
            print('Total Loss: %.4f' % np.mean(total_l[-100:]))
            print('classification Loss: %.4f' % np.mean(class_l[-100:]))
            print('Segmentation Loss: %.4f' % np.mean(seg_l[-100:]))
            print("Learning Rate: ", optimizer.param_groups[0]['lr'])
            if config.validate:
                dice = test_net(net, val_loader,val_num=config.val_num)
                save_model(net,dir_checkpoint + model_name + str(dice) + '_{}.pth'.format(i), save_model=save_cp)
                print("checkpoint saved")

            net.train()

    save_model(net,dir_checkpoint + model_name, save_model=save_cp)

def save_model(model, save_path, save_model=False):
    if save_model:
        torch.save(model.state_dict(), save_path)
        print('Checkpoint saved in {}!'.format(save_path))

def test_net(net,val_loader,val_num=100,dim=512):
    print("Starting Validation")
    # The device we are using, naturally, the gpu by default
    metric = Metric(device=device)
    # Keep track of the training loss, we need to save this later on
    val_accuracy, val_dice, f1_classes = metric.evaluate(net,val_loader,val_num,dim=512)
    print('Validation Dice: {0:.4g}  [===] Validation Accuracy: {1:.4g} [===] Validation Class F1: {2:.4g}'.format(val_dice, val_accuracy, f1_classes))
    return round(val_dice, 3)

if __name__ == '__main__':
    # Get the argumetns given through the terminal
    config = dense_unet_config()

    aux_params = dict(dropout=0.5, classes=config.n_classes)
    net = smp.Unet("efficientnet-b4", classes=config.n_classes, aux_params=aux_params, encoder_weights="imagenet")
    # If we would like to load a trained model
    print("GPU devices: ", torch.cuda.device_count())

    training_data = get_barrett_data(config)

    training_loader = DataLoader(training_data,
                             batch_size=config.batch_size,
                             shuffle=True,
                             drop_last=False)

    val_data = get_barrett_val_data(config)

    val_loader = DataLoader(val_data,
                             batch_size=1,
                             shuffle=False,
                             drop_last=False)

    net.train()
    net = nn.DataParallel(net)
    if config.model_checkpoint is not None:
        #checkpoint = torch.load(config.model_checkpoint)
        #net_state = net.state_dict()
        net.load_state_dict(torch.load(config.model_checkpoint))

    net.to(device)
    #cudnn.benchmark = True # faster convolutions, but more memory
    train_net(net, training_loader, val_loader, config, save_cp=True, model_name="efficientUnet_b4_10x_oversampling")

    print("Done Training")

