from typing import Literal
import torch
from torchvision.models import resnet18, resnet50
from torchvision.models.resnet import ResNet18_Weights, ResNet50_Weights
import torch.nn as nn
from tqdm import tqdm

import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 7
NUM_EPOCHS = 5

def create_resnet(
        size: Literal[18, 50], 
        num_classes=num_classes, 
        device=device,
        classifier = None,
    ):
    if size == 18:
        model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    elif size == 50:
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    else:
        raise ValueError(f"Unsupported ResNet size: {size}")
    
    if classifier is None:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        model.fc = classifier
    
    return model.to(device)

def run_epoch(model, loader, criterion, optimizer=None, device=device):
    if optimizer is None:
        model.eval()
        torch.set_grad_enabled(False)
    else:
        model.train()
        torch.set_grad_enabled(True)

    epoch_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader):
        images = images.to(device)
        labels = labels.to(device).squeeze(1).long() # # [B,1] -> [B]

        if optimizer is not None:
            optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        if optimizer is not None:
            loss.backward()
            optimizer.step()

        epoch_loss += loss.item() * images.size(dim=0)
        pred_labels = outputs.argmax(dim=1)
        correct += (pred_labels == labels).sum().item()
        total += labels.size(0)

    
    avg_loss = epoch_loss / len(loader.dataset)
    avg_acc = correct / total if total > 0 else 0

    torch.set_grad_enabled(True)

    return avg_loss, avg_acc 

def train_model(model, train_loader, val_loader, optimizer, criterion, num_epochs=NUM_EPOCHS, device=device):
    train_history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    for epoch in range(num_epochs):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device)

        train_history["train_loss"].append(train_loss)
        train_history["train_acc"].append(train_acc)
        train_history["val_loss"].append(val_loss)
        train_history["val_acc"].append(val_acc)

        print(f"Epoch {epoch} |"
              f"Train loss: {train_loss:.4f}, acc: {train_acc:.3f} |"
              f"Val loss: {val_loss:.4f}, acc: {val_acc:.3f}")
        
    return train_history

def plot_train_hist(train_history, type: Literal["loss", "acc"]):
    train_col = f"train_{type}"
    val_col = f"val_{type}"

    epochs = range(1, len(train_history[train_col]) + 1)
    plt.figure()

    plt.plot(epochs, train_history[train_col], label="Train")
    plt.plot(epochs, train_history[val_col], label="Validation")

    plt.xlabel("Epoch")
    plt.ylabel(type.upper())
    plt.legend()
    plt.title(f"Training vs. Validation {type.upper()}")

    return plt
