import torch
import torch.nn as nn
from models import SetTransformer, SmallSetTransformer, DeepSet, DeepSetTransformer 
import numpy as np
import matplotlib.pyplot as plt
from data.load_data import *
from torch.utils.data import DataLoader
from config import *
import torch.optim.lr_scheduler as lr_scheduler
from utils import test

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(config, model, training_loader, validation_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    # TODO: add classweights
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1,1,1,1]).float().to(device))
    #scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, total_steps=config.epochs)
    #scheduler = lr_scheduler.MultiplicativeLR(optimizer,lr_lambda=lambda epoch: 0.995)
    losses = []
    best_accuracy = 0
    best_accuracy_epoch = 0
    for epoch in range(config.epochs):
        for i, (x, y, idx) in enumerate(training_loader):

            x, y = x.to(device).permute(0,2,1), y.to(device).long()
            output = model(x)
            loss = criterion(output.squeeze(1), y.squeeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print("[==========================]")
        print("Current epoch: {}".format(epoch+1))
        print('Total Loss: %.4f' % np.mean(losses[-100:]))
        print("Learning Rate: ", optimizer.param_groups[0]['lr'])
        #scheduler.step()
        if config.validate and epoch%1==0: 
            accuracy, labels, predictions = test(config, model, validation_loader, epoch)
            if accuracy >= 0.65:
                best_accuracy = accuracy
                best_accuracy_epoch = epoch
                print(labels)
                print(predictions)
                torch.save(model.state_dict(), "../../Models/set_transformers/{}_{}_{}".format(epoch, round(best_accuracy, 3), "train_20x"))
            print("Accuracy: ", round(accuracy, 3))

        model.train()
    print("Best Accuracy After Training: ", round(best_accuracy, 3))
    return losses

def main():

    config = set_transformer_config("testing phase")

    # CONFIG!!!

    model = SetTransformer(set_size=config.set_size, hidden_dim=512, num_heads=32)

    # Load Data
    training_data = get_transformer_data(config)
    val_data = get_transformer_val_data(config)

    training_loader = DataLoader(training_data,
                             batch_size=config.batch_size,
                             shuffle=True,
                             drop_last=False)
    
    validation_loader = DataLoader(val_data,
                             batch_size=1,
                             shuffle=False,
                             drop_last=False)

    model = nn.DataParallel(model).to(device)
    if config.model_checkpoint is not None:
        with open(config.model_checkpoint, 'rb') as f:
                state_dict = torch.load(f)
                model.load_state_dict(state_dict)
                print("Model Loaded!")
            
    model.train()
    losses = train(config, model, training_loader, validation_loader)
    plt.plot(losses, label="Set Transformer")
    plt.legend()
    plt.xlabel("Steps")
    plt.ylabel("Mean Absolute Error")
    plt.yscale("log")
    plt.show()

if __name__ == '__main__':
    main()
