
import torch
import torchvision
import pandas as pd
from torchtnt.utils import get_module_summary
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm.auto import tqdm
from cjm_pandas_utils.core import markdown_to_pandas, convert_to_numeric, convert_to_string
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchvision.transforms.functional as F


def model_summarize(model, device):

    test_inp = torch.randn(1, 3, 256, 256).to(device)

    summary_df = markdown_to_pandas(f"{get_module_summary(model.eval(), [test_inp])}")

    # # Filter the summary to only contain Conv2d layers and the model
    summary_df = summary_df[summary_df.index == 0]

    return summary_df.drop(['In size', 'Out size', 'Contains Uninitialized Parameters?'], axis=1)


# the following fx performs a single pass through the training set

def train(model, train_dataloader, test_dataloader, epochs, bs, device, optimizer):
    
    """
    main training loop, iterates over epochs and over batches
    IN: model, train_dataloader, test_dataloader, epochs, batch size, device
    Returs: trained model, prints results for one epoch

    """
    
    all_train_losses = []
    all_val_losses = []

    flag = False

    for epoch in range(epochs):
        train_epoch_loss = 0
        val_epoch_loss = 0
    
        # put the model into training mode
        model.train()    
    
        #TRAINING LOOP
    
        # dt is for training the dataloader, 
        for i , dt in enumerate(train_dataloader):
            
            
            #imgs to device
            b_len = len(dt)
            imgs = []
            for i in range(b_len):
                imgs.append(dt[i][0].to(device))
            
            #imgs = [dt[0][0].to(device) , dt[1][0].to(device)]
        
            # targets list will contain dictionaries where the values from targ are moved to a specified device
                # Iterates over each element (t) in the targ list
                # Creates a dictionary by iterating over the key-value pairs in t 
                
            targ = []
            b_len = len(dt)
            for i in range(b_len):
                targ.append(dt[i][1])
                
                
            #targ = [dt[0][1] , dt[1][1]]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targ]
        
            # CALCULATE LOSS: for boxes, masks and the region proposal network
            loss = model(imgs, targets)
        
            # flag ensures that the initial parameters of the loss are only printed once
            if not flag:
                print(loss)
                flag = True
        
            # sum all the losses
            losses = sum([l for l in loss.values()])
        
            # add the losses into train_epoch_loss for one epoch
            train_epoch_loss += losses.cpu().detach().numpy()
        
            # optimizer zero
            optimizer.zero_grad()
        
            # backpropagation
            losses.backward()
        
            # optimize weights and biases
            optimizer.step()
        
        # summrize all losses
        all_train_losses.append(train_epoch_loss)
    
        #VALIDATION LOOP
    
        # put model into inference mode
        with torch.no_grad():
        
            for j , dt in enumerate(test_dataloader):
            
                #imgs to device
                
                imgs = []
                b_len = len(dt)
                for i in range(b_len):
                    imgs.append(dt[i][0].to(device))
                                
            
                # targets list will contain dictionaries where the values from targ are moved to a specified device
                    # Iterates over each element (t) in the targ list
                    # Creates a dictionary by iterating over the key-value pairs in t     
                    targ = []
                    b_len = len(dt)
                    for i in range(b_len):
                        targ.append(dt[i][1])
                                
                targets = [{k: v.to(device) for k, v in t.items()} for t in targ]
            
                # CALCULATE LOSS: for boxes, masks and the region proposal network
                loss = model(imgs , targets)
                losses = sum([l for l in loss.values()])
            
                # add the losses into val_epoch_loss for one epoch
                val_epoch_loss += losses.cpu().detach().numpy()
            
            # summrize all losses
            all_val_losses.append(val_epoch_loss)
      
    
        #print results for on epoch
        print(f"epoch: {epoch}    training loss {train_epoch_loss}    validation loss: {val_epoch_loss}")
    
    return all_val_losses, all_train_losses
