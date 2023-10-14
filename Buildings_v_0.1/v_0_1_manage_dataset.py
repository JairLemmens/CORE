
import random

def concatinate_data(images, boxes, masks, labels): 
    # IN: image tensors, box tensors, binary mask tensors, box labels tensor
    # OUT: list of the image tensor and dictionary of target values at each index
    keys = ['boxes', 'masks', 'labels']
    dataset_targets = []
    for b, l, m in zip(boxes, labels, masks):
    
        data_dict = {
            'boxes': b,
            'labels': l,
            'masks': m
        }
        dataset_targets.append(data_dict)
        
    dataset_fin = []

    for i in range(len(dataset_targets)):
        # Create a tuple with the dictionary and the array
        combined_data = (images[i], dataset_targets[i])
        dataset_fin.append(combined_data)    
    
    return dataset_fin


def test_train_split(dataset):
    # Shuffle the list randomly
    random.shuffle(dataset)
    dataset_sh = dataset.copy()

    split_index = int(0.75 * len(dataset_sh))

    # Split the list
    train_dataset = dataset_sh[:split_index]
    test_dataset = dataset_sh[split_index:]
    return train_dataset, test_dataset
