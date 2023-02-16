import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os


def analysis(dataset_name):
    global_explanation = np.zeros((30))
    mask = np.zeros((5000, 30))
    file_count = 0
    mask_dir = dataset_name + "/perforation_masks_train3/"
    for file_name in os.listdir(mask_dir):
        mask_file_path = mask_dir + file_name
        mask_data = np.load(mask_file_path, allow_pickle=True).item()
        mask = mask_data['mask']
        labels = mask_data['labels']
        explanation = list(mask_data['explanation'].values())
        explanation = np.array(explanation)
        if np.max(explanation) == np.min(explanation):
            continue
        explanation = (explanation-np.min(explanation))/(np.max(explanation)-np.min(explanation))
        
        for i in range(30):
            global_explanation[i] += explanation[i]
        file_count += 1
    global_explanation /= file_count
    indices = np.argsort(global_explanation)[::-1]
    global_explanation_dict = {indices[i]: (20-i) for i in range(20)}

    print(global_explanation)
    short_indices = []
    for i in range(len(indices)):
        if indices[i] < 10:
            short_indices.append(indices[i])
    short_indices_2 = []
    for i in range(len(indices)):
        if indices[i] < 15:
            short_indices_2.append(indices[i])
    print(short_indices, short_indices_2, indices)
    print(global_explanation_dict)
    print(file_count)
    
    s = "["
    for i in range(len(indices)):
        s += str(indices[i])
        s += ","
    s += "]"
    print(s)
    
#analysis("Urban")
analysis("ESC-kitchen")
#analysis("ESC-indoor")
#analysis("ESC-animal")