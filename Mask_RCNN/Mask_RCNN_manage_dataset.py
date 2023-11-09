
import random
from torch.utils.data import Dataset, DataLoader
from torchtnt.utils import get_module_summary
import torch
import torchvision


def indexes_to_cull(boxes, masks_binary, labels):
    """
    finds mismathed number of masks, boxes or labels for each index
    """
    indexes_to_cull = []

    for index, (b, m, l) in enumerate(zip(boxes, masks_binary, labels)):
        if len(b) != len(m) or len(m) != len(l) or len(b) != len(l):
            indexes_to_cull.append(index)
                    
    return indexes_to_cull
    

def test_train_split(objects):
    """ 
    Split the list for training and inference 
    """
    split_index = int(0.80 * len(objects))
    train_dataset = objects[:split_index]
    test_dataset = objects[split_index:]
    return train_dataset, test_dataset




class CustData(torch.utils.data.Dataset):
    
    """
    object CustData inherits from the torch.utils.data.Dataset Class
    
    in: images, masks, boxes, labels, image_ids
    returns: list of the image tensor and dictionary of target values at each index
    """
    def __init__(self, images, masks, boxes, labels, image_ids):
        self.images = images
        self.masks = masks
        self.boxes = boxes
        self.labels = labels
        self.image_ids = image_ids
        
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, index):
        
        # Retrieve the key for the image at the specified index
        image_ids = self.image_ids[index]
        mask = self.masks[index]
        box = self.boxes[index]
        label = self.labels[index]
        image = self.images[index]       
        
        
        return image, {'boxes': box,'labels': label, 'masks': mask} 
