"""
This file contains classes and helper functions used to load datasets to be used for training. It 
also contains helper functions for exploring and visualizing raw data. It does not contain Grad-CAM
related functionality, as that is stored in `gradcam.py`.
"""
import torch
from medimeta import MedIMeta
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import pandas as pd

# AI Usage: asked AI how to apply transforms to MedIMeta data; used the code it gave me to help
# write this
# AI Usage: Used AI to diagnose issue with label mismatch and to generate code to account for it
medimeta_to_derma_idx = {
    0: 4,  # Melanoma
    1: 5,  # Melanocytic nevus
    2: 1,  # Basal cell carcinoma
    3: 0,  # Actinic keratosis / Bowen’s disease
    4: 2,  # Benign keratosis
    5: 3,  # Dermatofibroma
    6: 6,  # Vascular lesion
}

class TransformedMedIMeta:
    def __init__(self, path, dataset, task, transform=None, label_map=medimeta_to_derma_idx):
        self.base_data = MedIMeta(path, dataset, task)
        self.transform = transform
        self.label_map = label_map

    def __len__(self):
        return len(self.base_data)
    
    def __getitem__(self, key):
        image, label = self.base_data[key]
        label = int(label)
        if self.label_map is not None:
            label = self.label_map[label]
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)
    
def get_class_names(info):
    """
    Get class names from info
    """

    n_classes = len(info["label"]) 
    class_names = [info["label"][str(i)] for i in range(n_classes)]
    return class_names

def count_datasets(datasets, info, n_classes):
    """
    Count number of images per class in a list of datasets

    :param datasets: A list of DermaMNIST datasets
    :returns: DataFrame with the dataset counts
    """
    class_names = get_class_names(info)
    dataset_names = ["train", "validation", "test"]

    total_counter = Counter()
    split_counters = {}

    for i in range(3):
        count_labels = [int(label[0]) for _, label in datasets[i]]
        class_counts = Counter(count_labels)
        
        split_name = dataset_names[i]
        split_counters[split_name] = class_counts
        
        total_counter += class_counts

        print(f"[{split_name}] number of images per class:")
        print("-----")
        for j in range(n_classes):   
            count = class_counts[j]
            print(f"{class_names[j]}: {count}")
        print("-----")

    total_images = sum(total_counter.values())
    print("[total] number and proportions of images per class:")
    print("-----")
    for j in range(n_classes):
        count = total_counter[j]
        proportion = count / total_images
        print(f"{class_names[j]}: {count} ({proportion:.2%})")
    print("-----")

    rows = []
    for j in range(n_classes):
        row = {
            "class": class_names[j],
            "train": split_counters["train"][j],
            "validation": split_counters["validation"][j],
            "test": split_counters["test"][j],
            "total": total_counter[j],
            "proportion": total_counter[j] / total_images
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    return df

def show_one_per_class(dataset, class_names):
    n_classes = len(class_names)
    fig, axes = plt.subplots(1, n_classes, figsize=(2*n_classes, 2))

    if n_classes == 1:
        axes = [axes]
    
    shown_classes = set()
    for img, label in dataset:
        class_idx = int(label[0])
        
        if class_idx not in shown_classes:
            ax = axes[class_idx]
            
            if hasattr(img, "numpy"): 
                img_np = img.permute(1, 2, 0).numpy()
            else:  
                img_np = np.array(img)
            
            ax.imshow(img_np)
            ax.set_title(class_names[class_idx], fontsize=8)
            ax.axis("off")
            shown_classes.add(class_idx)
        
        if len(shown_classes) == n_classes:
            break

    plt.tight_layout()
    plt.show()
