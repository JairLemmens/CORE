
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
from tqdm.auto import tqdm
import skimage as ski


import torch
import torchvision
from torch.amp import autocast # allows for differentr datatypes (torch.32/ torch.16)
from torchvision import transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torchvision.transforms.functional as F
from torchvision.ops import masks_to_boxes
from torch.utils.data import Dataset, DataLoader
from torchtnt.utils import get_module_summary
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks

from cjm_pytorch_utils.core import pil_to_tensor, tensor_to_pil, get_torch_device
from cjm_psl_utils.core import download_file, file_extract, get_source_code
from cjm_pandas_utils.core import markdown_to_pandas

# Import Mask R-CNN
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN, maskrcnn_resnet50_fpn
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, MaskRCNN_ResNet50_FPN_Weights


""" 
load images and apply transform to the images (normalise), out: image torch tensors
"""
def load_images_from_folder(images_dir):
    images = []
    transform = T.Compose([T.ToTensor()])
    for filename in os.listdir(images_dir):
        if filename.endswith('.jpeg'):
            img = Image.open(os.path.join(images_dir, filename))
            img_t = transform(img)
            images.append(img_t)
    return images

""" 
generate indexes for the images
"""
def generate_index_list(image_tensors):
    num_images = len(image_tensors)
    return list(range(num_images))


"""
load masks, transform to numpy arrays 
"""
def load_masks_from_folder(mask_dir):
    masks = []
    for filename in os.listdir(mask_dir):
        if filename.endswith('.jpeg'):
            mask = Image.open(os.path.join(mask_dir, filename))
            mask_arr = np.array(mask)
            masks.append(mask_arr)
    return masks
