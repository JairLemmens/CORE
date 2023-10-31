import random
from matplotlib import image
import torch
import os
import numpy as np
from torch.utils.data import TensorDataset

def load_data(sample_dir,mask_dir, pool_size =100000):

    #dataloading
    samples = []
    masks = []
    _filenames = os.listdir(sample_dir)
    if len(_filenames) > pool_size:
        _filenames = random.sample(_filenames,pool_size)

    for _filename in _filenames:
        _filename = os.path.splitext(_filename)[0]
        _mask = image.imread(f'{mask_dir}/{_filename}.jpeg').astype("float32")
        _img = image.imread(f'{sample_dir}/{_filename}.jpeg').astype("float32")
        samples.append(_img/255)
        masks.append(_mask/255)
    samples = torch.tensor(np.array(samples,dtype='float32'))
    masks = torch.tensor(np.array(masks,dtype='float32')).swapaxes(-1,-3)
    
    return(TensorDataset(samples,masks))