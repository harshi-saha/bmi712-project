"""
This file contains helper functions for running Grad-CAM.
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
import torch.nn as nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from .device import get_device

device = get_device()

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

# AI Usage: asked ChatGPT to explain how to modify this
# function to show the original images above the GradCAM ones for easier visualization
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
