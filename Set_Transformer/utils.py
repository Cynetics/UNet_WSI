import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from config import *
import torch.optim.lr_scheduler as lr_scheduler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test(config, model, validation_loader, epoch=0):
    model.train()
    runs = 1
    predictions = []
    soft_predictions = torch.zeros((len(validation_loader),runs,4))
    labels = []
    origins = []
    for i in range(runs):
        p = []
        l = []
        for j, (x, y, origin) in enumerate(validation_loader):

            x, y = x.to(device).permute(0,2,1), y.to(device).long().squeeze(1)
            output = model(x).squeeze(1)
            soft_predictions[j, i, :] = torch.softmax(output, dim=1)
            output = torch.argmax(output, dim=1)
            p += list(output.cpu().numpy())
            l += list(y.cpu().numpy())
            
            if not origin[0] in origins:
                origins.append(origin[0])
        predictions.append(p)
        labels.append(l)
    
    predictions = torch.mode(torch.tensor(predictions),dim=0)[0]
    labels = torch.mode(torch.tensor(labels),dim=0)[0]
    accuracy = (predictions==labels).float().mean().item()
    soft_preds = torch.mean(soft_predictions,dim=1)
    var = torch.var(soft_predictions,dim=1)

    print("predictions: ", predictions)
    print("labels: ", labels)
    
    # logging
    if accuracy > 0.651:
        with open("results_set_full20x_wsi_sorted_2{}_{}.csv".format(epoch, round(accuracy, 3)), "a") as f:
            for i in range(len(labels)):
                f.write("{}, {}, {}, {}, {}, {}, {}\n".format(origins[i][:-3], labels[i], predictions[i], soft_preds[i, 0], soft_preds[i,1], soft_preds[i,2], soft_preds[i,3]))
    return accuracy, labels, predictions