
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import datetime
from functools import partial
from glob import glob
import math
import multiprocessing
from pathlib import Path
import random


import torch
import torchvision
from torch.amp import autocast # allows for differentr datatypes (torch.32/ torch.16)
from torchvision import transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchvision.transforms.functional as F


# load images and apply transform to the images (normalise), out: image torch tensors
def load_images_from_folder(images_dir):
    images = []
    transform = T.Compose([T.ToTensor()])
    for filename in os.listdir(images_dir):
        if filename.endswith('.jpeg'):
            img = Image.open(os.path.join(images_dir, filename))
            img_t = transform(img)
            images.append(img_t)
    return images

# generate indexes for the images
def generate_index_list(image_tensors):
    num_images = len(image_tensors)
    return list(range(num_images))


# load masks, transform to numpy arrays 
def load_masks_from_folder(mask_dir):
    masks = []
    for filename in os.listdir(mask_dir):
        if filename.endswith('.jpeg'):
            mask = Image.open(os.path.join(mask_dir, filename))
            mask_arr = np.array(mask)
            masks.append(mask_arr)
    return masks
