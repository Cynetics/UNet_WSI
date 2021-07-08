import torch
import numpy as np
from PIL import Image
import os
from sklearn.metrics import classification_report
np.set_printoptions(threshold=np.nan)

class Metric(object):
    """
    Various metrics, including the dice coefficient for, for individual examples. 
    This method currently does not take multi-class into account or binary one-hot vectors 
    for that matter. We need to change it as soon as possible.
    """
    def __init__(self, compute_jaccard=False, device="cuda"):
        self.device = device
        
    def multi_dice(self, input, target, n_labels):
        # Compute the dice per class 
        # and take the mean over all the classes save for the background dice
        dices = []
        for i in range(n_labels):
            dice = self.dice((input==i).float(), (target==i).float())
            dices.append(dice.item())
        print(dices)
        return np.mean(dices[1:])
        
    def dice(self, input, target):
        ''' 
        Given an input and target compute the dice score
        between both. Dice: 1 - (2 * |A U B| / (|A| + |B|))
        '''
        eps = 1e-6
        if len(input.shape) > 1:
            input, target = input.view(-1), target.view(-1)

        inter = torch.dot(input, target)
        union = torch.sum(input) + torch.sum(target) + eps
        dice = (2 * inter.float() + eps) / union.float()
        return dice

    def pixel_wise(self, input, target):
        """
        Regular pixel_wise accuracy metric, we just
        compare the number of positives and divide it
        by the number of pixels.
        """
        # Flatten the matrices to make it comparable
        input = input.view(-1)
        target = target.view(-1)
        correct = torch.sum(input==target)
        return (correct.item() / len(input))
    
    def evaluate(self, net, val_loader, N_val, dim=512, binary=False):
        
        """"
        Given the trained network, and the validation set, compute the dice score.
        """
        print("Initiated Metric Evaluation")

        net.eval()
        preds_real, masks_real = torch.zeros(N_val, dim**2), torch.zeros(N_val, dim**2)
        preds_class, real_class = torch.zeros(N_val), torch.zeros(N_val)
        with torch.no_grad():
            for i, (img, true_masks) in enumerate(val_loader):
                
                #img, true_masks = next(iter(val_loader))
                img = img.to(self.device)#.permute(0,3,1,2)
                true_masks = true_masks.to(self.device)

                B = true_masks.shape[0]
                patch_labels = torch.zeros(B).to(self.device)
                labels = [0,1,3,2]
                #labels = [0,1,2,3,4,5]
                for label in labels:
                    hard_labels = (true_masks.view(B, -1)==label).long()
                    hard_labels = hard_labels.sum(dim=1)
                    
                    patch_labels[torch.where(hard_labels > (512*512*0.2))] = label
                
                predictions, class_predictions = net(img)
                class_predictions = (torch.argmax(class_predictions, dim=1)).float()
                predictions = (torch.argmax(predictions, dim=1)).float()
                preds_real[i,:] = predictions.view(-1)
                masks_real[i,:] = (true_masks.view(-1)).float()
                preds_class[i] = class_predictions
                real_class[i] = patch_labels
                
        preds, masks = preds_real.view(-1), masks_real.view(-1)
        n_labels = 4
        #n_labels = 6
        seg_dice = self.multi_dice(preds, masks, n_labels)
        acc = self.pixel_wise(preds, masks)
        
        f1_classes = self.multi_dice(preds_class, real_class, n_labels)

        return acc, seg_dice, f1_classes
            
            