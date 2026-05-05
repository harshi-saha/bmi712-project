import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import models
import torch.nn as nn
from medmnist import Evaluator
import torch.optim as optim
from collections import Counter
import pandas as pd
from torchvision.models.resnet import ResNet18_Weights, ResNet50_Weights
from sklearn.metrics import roc_auc_score
from medimeta import MedIMeta

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import torch.nn.functional as F
import matplotlib as mpl

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

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


def get_resnet(type, n_classes=7, device=device):
    if type == 18:
        model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    elif type == 50:
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    else:
        raise ValueError(f"Unsupported ResNet type: {type}")
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model.to(device)

def evaluate(model, loader, device=device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.squeeze().long().to(device)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total

# AI Usage: Asked ChatGPT how to make this funciton work on any image
# size from medmnist
def compute_auc(model, loader, split, size, device=device):
    model.eval()
    
    y_true = []
    y_score = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.squeeze().long().to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            y_true.append(labels.cpu())
            y_score.append(probs.cpu())

    y_true = torch.cat(y_true).numpy()
    y_score = torch.cat(y_score).numpy()

    evaluator = Evaluator("dermamnist", split, size=size)
    metrics = evaluator.evaluate(y_score)

    return metrics

# AI Usage: AI generated function based on the above compute_auc function
def compute_auc_multiclass(model, loader, device=device, average="macro"):
    """
    Computes multi-class ROC AUC using one-vs-rest scheme.
    average: "macro", "weighted", or None (see sklearn.roc_auc_score docs)
    """
    model.eval()
    y_true = []
    y_score = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.squeeze().long().to(device)

            outputs = model(images)                  # (B, C)
            probs = torch.softmax(outputs, dim=1)    # class probabilities

            y_true.append(labels.cpu())
            y_score.append(probs.cpu())

    y_true = torch.cat(y_true).numpy()      # shape: (N,)
    y_score = torch.cat(y_score).numpy()    # shape: (N, C)

    # multi_class="ovr" is typical, "ovo" also available
    auc = roc_auc_score(
        y_true, 
        y_score, 
        multi_class="ovr", 
        average=average
    )
    return auc

# AI Usage: asked ChatGPT how to incorperate the learning rate into this function
def train_model(model, train_loader, val_loader, epochs=5, lr=1e-3, device=device, criterion=nn.CrossEntropyLoss()):
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        running_loss = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.squeeze().long().to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        val_acc = evaluate(model, val_loader)

        print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss:.4f} | Val Acc: {val_acc:.4f}")

    return model


# https://github.com/Peachypie98/CBAM

class CAM(nn.Module):
    def __init__(self, channels, r=16):
        super().__init__()

        self.linear = nn.Sequential(
            nn.Linear(channels, channels // r),
            nn.ReLU(inplace=True),
            nn.Linear(channels // r, channels)
        )

    def forward(self, x):
        max_pool = F.adaptive_max_pool2d(x, 1)
        avg_pool = F.adaptive_avg_pool2d(x, 1)

        b, c, _, _ = x.size()

        max_out = self.linear(max_pool.view(b, c)).view(b, c, 1, 1)
        avg_out = self.linear(avg_pool.view(b, c)).view(b, c, 1, 1)

        attention = torch.sigmoid(max_out + avg_out)
        return attention * x


class SAM(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)

        concat = torch.cat([max_pool, avg_pool], dim=1)
        attention = torch.sigmoid(self.conv(concat))

        return attention * x


class CBAM(nn.Module):
    def __init__(self, channels, r=16):
        super().__init__()
        self.cam = CAM(channels, r)
        self.sam = SAM()

    def forward(self, x):
        out = self.cam(x)
        out = self.sam(out)
        return out + x
    
class ResNet_CBAM(nn.Module):
    def __init__(self, model_name, n_classes):
        super().__init__()

        if model_name == "resnet18":
            self.backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        elif model_name == "resnet50":
            self.backbone = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        num_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.attention = CBAM(num_features)
        self.classifier = nn.Linear(num_features, n_classes)

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        x = self.attention(x)

        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
# AI Usage: asked ChatGPT to explain how to modify this
# function to show the original images above the GradCAM ones for easier visualization
# def run_gradcam_grid(model, dataset, indices):
#     model.eval()
#     target_layer = model.backbone.layer4[-1]
#     cam = GradCAM(model=model, target_layers=[target_layer])

#     fig, axes = plt.subplots(2, len(indices), figsize=(5 * len(indices), 10))
#     if len(indices) == 1:
#         axes = axes.reshape(2, 1)

#     for col, idx in enumerate(indices):
#         img, label = dataset[idx]
#         input_tensor = img.unsqueeze(0).to(device)
#         with torch.no_grad():
#             output = model(input_tensor)
#             pred = output.argmax(dim=1).item()

#         grayscale_cam = cam(
#             input_tensor=input_tensor,
#             targets=[ClassifierOutputTarget(pred)]
#         )[0]

#         rgb_img = img.permute(1, 2, 0).cpu().numpy()
#         rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())
#         visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

#         ax_orig = axes[0, col]
#         ax_orig.imshow(rgb_img)
#         ax_orig.set_title(f"Idx {idx}\nT:{label} P:{pred}")
#         ax_orig.axis("off")

#         ax_cam = axes[1, col]
#         ax_cam.imshow(visualization)
#         ax_cam.axis("off")

#     fig.subplots_adjust(right=0.88)
#     cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])

#     norm = mpl.colors.Normalize(vmin=0, vmax=1)
#     sm = mpl.cm.ScalarMappable(cmap="jet", norm=norm)
#     sm.set_array([])
#     cbar = fig.colorbar(sm, cax=cbar_ax)
#     cbar.set_label("Attention intensity (Grad-CAM)", rotation=90, labelpad=12)

#     plt.show()

def get_default_target_layer(model: nn.Module):
    """
    Try to get a reasonable default target layer for Grad-CAM
    that works for both plain ResNets and ResNets wrapped in a backbone.
    """
    # If model has a .backbone attribute (like your CBAM models), use that; otherwise use model itself.
    backbone = getattr(model, "backbone", model)

    # Typical torchvision ResNet, or your CBAM backbone
    if hasattr(backbone, "layer4"):
        # Last block in layer4 (usually bottleneck/block containing last conv)
        return backbone.layer4[-1]

    # Fallback: last Conv2d in the whole model
    conv_layers = [m for m in backbone.modules() if isinstance(m, nn.Conv2d)]
    if len(conv_layers) == 0:
        raise RuntimeError("No Conv2d layers found in model; cannot select target layer for Grad-CAM.")
    return conv_layers[-1]


def run_gradcam_grid(model, dataset, indices, target_layer=None):
    model.eval()
    model.to(device)

    # If no target_layer passed, try to infer one that works for ResNet / ResNet+CBAM
    if target_layer is None:
        target_layer = get_default_target_layer(model)

    cam = GradCAM(model=model, target_layers=[target_layer])

    fig, axes = plt.subplots(2, len(indices), figsize=(5 * len(indices), 10))
    if len(indices) == 1:
        axes = axes.reshape(2, 1)

    for col, idx in enumerate(indices):
        img, label = dataset[idx]
        input_tensor = img.unsqueeze(0).to(device)

        # This is only for prediction; GradCAM will do its own forward with gradients
        with torch.no_grad():
            output = model(input_tensor)
            pred = output.argmax(dim=1).item()

        grayscale_cam = cam(
            input_tensor=input_tensor,
            targets=[ClassifierOutputTarget(pred)]
        )[0]

        rgb_img = img.permute(1, 2, 0).cpu().numpy()
        rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min() + 1e-8)
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        ax_orig = axes[0, col]
        ax_orig.imshow(rgb_img)
        ax_orig.set_title(f"Idx {idx}\nT:{label} P:{pred}")
        ax_orig.axis("off")

        ax_cam = axes[1, col]
        ax_cam.imshow(visualization)
        ax_cam.axis("off")

    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])

    norm = mpl.colors.Normalize(vmin=0, vmax=1)
    sm = mpl.cm.ScalarMappable(cmap="jet", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Attention intensity (Grad-CAM)", rotation=90, labelpad=12)

    plt.show()

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
