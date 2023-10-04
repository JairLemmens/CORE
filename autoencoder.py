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



name = 'AutoEncoder_03_10'
depths=[3, 3, 9, 3,3]
dims=[96, 192, 384, 768,768]
device = 'cuda'


##AUTOENCODER DEFINITION
class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(device=device,depths=depths,dims = dims)
        self.decoder = Decoder(device=device,depths=depths,dims = dims)

    def forward(self,x):
        encoding = self.encoder(x)
        decoding = self.decoder(encoding)
        return(decoding)



##DATALOADER
labels = []
data = []
for filename in os.listdir('./Data/AE_Training'):
    array = image.imread(f'./Data/AE_Training/{filename}')
    array = np.moveaxis(array,-1,0)
    
    array = np.divide(array, 255,dtype='float32')
    data.append(array)
    labels.append(0)
data = np.array(data)
print(data.shape)
dataset = TensorDataset(torch.tensor(data),torch.tensor(labels))
dataloader = DataLoader(dataset, batch_size=12, shuffle=True)
l = len(dataloader)



##TRAINING
model = AutoEncoder()
model = model.to(device)  
mse_loss = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 4e-3)
logger = SummaryWriter(os.path.join("runs", name))

for epoch in range(100):
    logging.info(f"Starting epoch {epoch}:")
    pbar = tqdm(dataloader)
    for i,(imgs,_) in enumerate(pbar):
        imgs = imgs.to(device)
        reconstruction = model(imgs)
        loss = mse_loss(reconstruction,imgs)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.set_postfix(MSE=loss.item())
        logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)
    torch.save(model.state_dict(), os.path.join("models", name, f"{epoch}ckpt.pt"))