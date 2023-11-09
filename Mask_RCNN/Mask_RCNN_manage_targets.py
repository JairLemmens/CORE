import numpy as np
import os
from PIL import Image
import torch
import torchvision
from torchvision import transforms as T
import cv2
import skimage as ski
import torchvision.transforms.functional as F
from torchvision.utils import draw_bounding_boxes
from torchvision.ops import masks_to_boxes

def mask_to_2dim(mask_arrays):
    
    # takes a lists of 3 dim numpy arrays, 
    # returns 2dim torch tensor in form of 0 for background, 1 for undamaged, 2 for damaged
    masks_transformed = []
    

    # Define color thresholds for blue and magenta
    green_lower = np.array([0, 50, 0], dtype=np.uint8)
    green_upper = np.array([100, 255, 100], dtype=np.uint8)

    red_lower = np.array([100, 0, 0], dtype=np.uint8)
    red_upper = np.array([255, 70, 100], dtype=np.uint8)
    
    for mask_arr in mask_arrays: 
        
        
        # Create masks for blue and magenta regions
        green_mask = cv2.inRange(mask_arr, green_lower, green_upper)
        red_mask = cv2.inRange(mask_arr, red_lower, red_upper)

        # Combine the masks to get the final transformed mask
        transformed_mask = np.zeros_like(green_mask)
        transformed_mask[green_mask > 0] = 1  # Object 1 (Blue)
        transformed_mask[red_mask > 0] = 2  # Object 2 (Magenta)
        
        masks_transformed.append(transformed_mask)
        
    # transform to torch tensor
            
    masks_array = np.array(masks_transformed)
    masks_tensor = torch.from_numpy(masks_array)
    
    return masks_tensor



def mask_to_2dim_no_cv2(mask_arrays):
    masks_transformed = []
    
    for mask_arr in mask_arrays:
        # Define color thresholds for blue and magenta
        blue_lower = np.array([0, 0, 100], dtype=np.uint8)
        blue_upper = np.array([80, 80, 255], dtype=np.uint8)

        magenta_lower = np.array([120, 0, 120], dtype=np.uint8)
        magenta_upper = np.array([255, 100, 255], dtype=np.uint8)

        # Create masks for blue and magenta regions
        blue_mask = np.all((mask_arr >= blue_lower) & (mask_arr <= blue_upper), axis=-1)
        magenta_mask = np.all((mask_arr >= magenta_lower) & (mask_arr <= magenta_upper), axis=-1)

        # Create the transformed mask
        transformed_mask = np.zeros_like(blue_mask, dtype=np.uint8)
        transformed_mask[blue_mask] = 1  # Object 1 (Blue)
        transformed_mask[magenta_mask] = 2  # Object 2 (Magenta)
        
        masks_transformed.append(transformed_mask)
    
    # Concatenate the transformed masks and convert to torch tensor
    masks_tensor = torch.from_numpy(np.stack(masks_transformed, axis=0))
    
    return masks_tensor





def cull_if_no_bulding(masks_transformed, image_tensors):
    cull_indexes = []
    for i, tensor in enumerate(masks_transformed):
        # Check if all elements in the tensor are zeros
        if torch.all(tensor == 0):
            cull_indexes.append(i)
            
    masks_culled = [mask for i, mask in enumerate(masks_transformed) if i not in cull_indexes]
    image_culled = [image for i, image in enumerate(image_tensors) if i not in cull_indexes]
    return masks_culled, image_culled


def masks_2_dim_to_booleans(masks_transformed):
    masks_binary = []
    
    # takes a list of 2dim torch tensor ans transformes into list of boolean arrays for each instance

    for mask in masks_transformed:
            # use Connected Component Analysis to extract all objects from the image
        
            mask_np, count = ski.measure.label(mask, connectivity=1, return_num=True)
            mask_test = torch.from_numpy(np.array(mask_np))

            # We get the unique colors, as these would be the object ids.
            mask_obj_ids = torch.unique(mask_test)

            # first id is the background, so remove it.
            mask_obj_ids = mask_obj_ids[1:]
        
            # split the color-encoded mask into a set of boolean masks.
            mask_boolean = mask_test == mask_obj_ids[:, None, None]
        
            masks_binary.append(mask_boolean)
    
    return masks_binary




def mask_to_box(masks_transformed):
    
    """
    args: 3 dim array of image masks 
    returns: tuple of bounding boxes, tuple of boxes labels
    """
    
    boxes = []
    labels = []

    for mask in masks_transformed:
        labels_mask = []
        
        ### we split the masks into damaged and andamaged tessors
        standing_list = []
        # We get the unique colors, as these would be the object ids.
        obj_ids = torch.unique(mask)
        # first id is the background, so remove it.
        obj_ids = obj_ids[1:]

        if len(obj_ids) == 1:
           
            # split the color-encoded mask into a set of boolean masks.
            standing = mask == obj_ids[:, None, None]
            standing_tensor = standing.int()
            
            
        elif len(obj_ids) == 2:
            separate_masks = mask == obj_ids[:, None, None]
            standing, collapsed = torch.split(separate_masks, 1, dim=0)
            standing_tensor, collapsed_tensor = standing.int(), collapsed.int()

        ### standing buildings
        
        # use Connected Component Analysis to extract all objects from the image
        standing_np, count = ski.measure.label(standing_tensor, connectivity=1, return_num=True)
        standing_test = torch.from_numpy(np.array(standing_np))
        
        # We get the unique colors, as these would be the object ids.
        standing_obj_ids = torch.unique(standing_test)
        
        # first id is the background, so remove it.
        standing_obj_ids = standing_obj_ids[1:]
        
        # split the color-encoded mask into a set of boolean masks.
        standing_boolean = standing_test == standing_obj_ids[:, None, None]        
        
        #make boxes (x1, x2, y1, y2)
        standing_boxes_test = masks_to_boxes(standing_boolean)
        
        #cull mini boxes
        
        cull_list = []
        for num, i in enumerate(standing_boxes_test):
            result_x = i[0] - i[2]
            result_y = i[1] - i[3]
            if result_x == 0 or result_y == 0:
                cull_list.append(num)
            
        cull_list.sort(reverse=True)
        
        for index in cull_list:
            standing_boxes_test = torch.cat((standing_boxes_test[:index], standing_boxes_test[index + 1:]))
        
        # output: boxes and labels
        label1 = 1
        standing_list = [(row) for row in standing_boxes_test]

        for i in range(len(standing_list)):
            labels_mask.append(label1)
            
        ### collapsed buildings
        collapsed_list = []

        # use Connected Component Analysis to extract all objects from the image
        
        if len(obj_ids) == 2:  # Need to handle this condition
            collapsed_np, count = ski.measure.label(collapsed_tensor, connectivity=1, return_num=True)
            
            collapsed_test = torch.from_numpy(np.array(collapsed_np))
            
            collapsed_obj_ids = torch.unique(collapsed_test)
            collapsed_obj_ids = collapsed_obj_ids[1:]
            
            collapsed_boolean = collapsed_test == collapsed_obj_ids[:, None, None]
            collapsed_boxes_test = masks_to_boxes(collapsed_boolean)
            
            #cull mini boxes
        
            cull_list = []
            for num, i in enumerate(collapsed_boxes_test):
                result_x = i[0] - i[2]
                result_y = i[1] - i[3]
                if result_x == 0 or result_y == 0:
                    cull_list.append(num)
            
            cull_list.sort(reverse=True)
        
            for index in cull_list:
                collapsed_boxes_test = torch.cat((collapsed_boxes_test[:index], collapsed_boxes_test[index + 1:]))
            
            # output: boxes and labels

            label2 = 2
            collapsed_list = [(row) for row in collapsed_boxes_test]
            for i in range(len(collapsed_list)):
                labels_mask.append(label2)
                
                

        both_lists = standing_list + collapsed_list
        boxes.append(both_lists)
        # make boxes to torch.int64 datatype

        boxes_int64 = []

        for box in boxes:
            tensor_box = torch.stack(box)
            boxes_int64.append(tensor_box)

        ### append labels list

        labels.append(labels_mask)

        labels_int64 = []
        for label in labels:
            tensor_label = torch.from_numpy(np.array(label)).to(torch.int64)
            labels_int64.append(tensor_label)

    return boxes_int64, labels_int64