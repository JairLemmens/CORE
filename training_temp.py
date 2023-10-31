import os
import torch
import torch.nn as nn
from transformer_modules_20_10 import ATM_ViT
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import logging
from utils import load_data

sample_dir = r"C:\Users\Admin\Documents\studentassistent-privé\Jair\CORE\Transformer_Training\sample"
mask_dir = r'C:\Users\Admin\Documents\studentassistent-privé\Jair\CORE\Transformer_Training\mask'
lr = 2e-4
patch_size=16
pool_size = 100000
num_layers = 9 
name = 'transformer_24_10'
device = 'cuda'



model = ATM_ViT(768,num_layers=num_layers).to(device)

#Training
optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay=.1)
logger = SummaryWriter(os.path.join("runs", name))
ce_loss = nn.CrossEntropyLoss()

for epoch in range(500):
    logging.info(f"Starting epoch {epoch}:")
    if epoch % 10 == 0:
        dataloader = DataLoader(load_data(sample_dir,mask_dir,pool_size), batch_size = 256, shuffle=True)
        l = len(dataloader)
    pbar = tqdm(dataloader)
    
    for i,(samples,masks) in enumerate(pbar):
        num_patches = masks.shape[-1]//patch_size
        
        samples = samples.to(device)
        masks = masks.to(device)
        sample_patches = samples.unfold(1,patch_size,patch_size).unfold(2,patch_size,patch_size).flatten(-3,-1).flatten(1,2)

        q,attns,class_prediction = model(sample_patches)
        
        fold = nn.Fold((num_patches,num_patches),kernel_size=1)

        total_attn = torch.sum(torch.stack(attns,dim=-1),dim=-1)
        mask = nn.functional.interpolate(fold(total_attn),size=(128,128), mode = 'bilinear',align_corners=False)
        semseg = torch.einsum('bq,bqhw->bqhw', class_prediction[..., :-1].mean(dim=1),mask.sigmoid())
        
        loss = ce_loss(semseg,masks[:,:2,:,:])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pbar.set_postfix(ce=loss.item())
        logger.add_scalar("CE", loss.item(), global_step=epoch * l + i)

    if epoch % 10 == 0:
        torch.save(model.state_dict(), os.path.join("models", name, f"{epoch}ATM_ViT_ckpt.pt"))