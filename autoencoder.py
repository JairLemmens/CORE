import os
import torch
import numpy as np
import torch.nn as nn
from nn_modules import Encoder,Decoder
from torch.utils.data import DataLoader, TensorDataset
import logging
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from matplotlib import image
logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

# dConv kernel size 7
# name = 'AutoEncoder_06_10_2'
# depths=[3, 3, 3, 9, 3, 3, 3]
# dims=[3, 6, 12, 24, 48,96,192]


#dConv kernel size 7
# name = 'AutoEncoder_06_10_3'
# depths=[3, 3, 3, 9]
# dims=[3, 6, 12, 24]


# name = 'AutoEncoder_06_10_4_noNorm'
# depths=[3, 3, 3, 9, 3, 3, 3]
# dims=[3, 6, 12, 24, 48,96,192]
# dConv_kernel_size=3

#added batch norm
# name = 'AutoEncoder_06_10_5'
# depths=[3, 3, 3, 9, 3, 3, 3]
# dims=[3, 6, 12, 24, 48,96,192]
# dConv_kernel_size=3

#lowered learning rate from 4e-3 to 4e-4 because of instability
# name = 'AutoEncoder_06_10_6'
# depths=[3, 3, 3, 9, 3, 3, 3]
# dims=[3, 6, 12, 24, 48,96,192]
# dConv_kernel_size=3

# kernel size to 5
# name = 'AutoEncoder_09_10'
# depths=[3, 3, 3, 9, 3, 3, 3]
# dims=[3, 6, 12, 24, 48,96,192]
# dConv_kernel_size=5

#predicting masks instead of imgs
name = 'AutoEncoder_10_10'
depths=[3, 3, 3, 9, 3, 3, 3]
dims=[3, 6, 12, 24, 48,96,192]
dConv_kernel_size=7



##AUTOENCODER DEFINITION
class AutoEncoder(nn.Module):
    def __init__(self,depths,dims, dConv_kernel_size=7):
        super().__init__()
        self.encoder = Encoder(depths=depths,dims = dims, dConv_kernel_size = dConv_kernel_size)
        self.decoder = Decoder(depths=depths,dims = dims, dConv_kernel_size = dConv_kernel_size)

    def forward(self,x):
        encoding = self.encoder(x)
        decoding = self.decoder(encoding)
        return(decoding)
    


##DATALOADER
labels = []
data = []
device = 'cuda'
for filename in os.listdir('./AE_debug/sample'):
    sample = image.imread(f'./AE_debug/sample/{filename}')
    sample = np.moveaxis(sample,-1,0)
    sample = np.divide(sample, 255,dtype='float32')
    data.append(sample)

    label = image.imread(f'./AE_debug/mask/{filename}')
    label = np.moveaxis(label,-1,0)
    label = np.divide(label, 255,dtype='float32')
    labels.append(label)
data = np.array(data)
labels = np.array(labels)
dataset = TensorDataset(torch.tensor(data),torch.tensor(labels))
dataloader = DataLoader(dataset, batch_size=96, shuffle=True)
l = len(dataloader)



##TRAINING
model = AutoEncoder(depths,dims,dConv_kernel_size)
model = model.to(device)  
mse_loss = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 4e-4)
#logger = SummaryWriter(os.path.join("runs", name))

for epoch in range(51):
    logging.info(f"Starting epoch {epoch}:")
    pbar = tqdm(dataloader)
    for i,(samples,labels) in enumerate(pbar):
        samples = samples.to(device)
        #labels = labels.to(device)
        reconstruction = model(samples)
        loss = mse_loss(reconstruction,samples)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.set_postfix(MSE=loss.item())
        #logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)
    
    #torch.save(model.state_dict(), os.path.join("models", name, f"{epoch}ckpt.pt"))