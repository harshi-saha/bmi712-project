"""
This file contains helpers for creating ResNet models
"""
from torchvision.models.resnet import ResNet18_Weights, ResNet50_Weights
from torchvision.models import resnet18, resnet50
import torch.nn as nn
from typing import Literal
from .device import get_device
from .attention import se_resnet18
from .constants import NUM_CLASSES

device = get_device()

def get_resnet(type: Literal[18, 50], n_classes: int=7, device: str=device):
    """
    A duplicate of `create_resnet` kept around for backward compatibility with legacy code. Please 
    use `create_resnet` instead as it has more functionality. This function just wraps that one.
    
    :param type: The size of the ResNet model. Accepted values are `18` or `50`.
    :param n_classes: The number of classes you will be predicting (multi-label, single-class)
    :param device: The device to send the model to when it is created.
    """
    return create_resnet(size=type, num_classes=n_classes, device=device)

def create_resnet(
        size: Literal[18, 50],
        num_classes: int=NUM_CLASSES,
        device: str=device,
        classifier: bool=None,
        use_se: bool = False,
    ):
    """
    Create a ResNet model with a classifier. Optionally add SE attention.

    :param size: The size of the ResNet model. Supported values are 18, for ResNet18 and 50 for ResNet50
    :param num_classes: The number of classes to use for predicting (multi-class single label).
    :param device: The device to put the model on.
    :param classifier: Allows user to pass a custom classifier layer (e.g., by passing `nn.Linear(...)`)
    for this paramter. Defaults to an `nn.Linear()` layer with `num_classes` classes.
    :param use_se: Whether to create a ResNet model using SE. Only supported for ResNet18. This parameter
    is ignored if `size == 50`.
    """
    if size == 18:
        if use_se:
            # use ImageNet weights if desired; can start with None to train from scratch
            model = se_resnet18(num_classes=num_classes,
                                weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    elif size == 50:
        # You could later define SE for ResNet50 similarly
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    else:
        raise ValueError(f"Unsupported ResNet size: {size}")

    if classifier is None:
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        model.fc = classifier

    return model.to(device)
