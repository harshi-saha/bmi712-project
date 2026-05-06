"""
This file contains helper functions for training models.

AI Usage: AI was used to help refactor some of the functions contained in here to get rid of duplicate
code and combine functions that were essentially doing the same thing, but with small differences in
behavior.
"""
import torch.optim as optim
import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score
from .device import get_device
from .constants import NUM_CLASSES

device = get_device()

# AI Usage: used AI to refactor this function from old ones
def _run_epoch_core(
    model,
    loader,
    criterion=None,
    optimizer=None,
    device=device,
    collect_probs=False,
):
    """
    Core epoch logic for both train and eval.

    :param model: The model on which to run the epoch.
    :param loader: The dataloader to use to run the epoch.
    :param criterion: The loss function for the model.
    :param optimizer: The optimizer for the model. Leave as None to run an epoch when evaluating.
    :param device: Device to run the epoch on.
    :param collect_probs: Whether to collect per-sample information for computing per-sample AUC
    :returns: A Dictionary: 
    ```
    {
        "loss_sum": float,
        "loss_avg": float,
        "correct": int,
        "total": int,
        "y_true": np.ndarray or None,
        "y_prob": np.ndarray or None,
    }
    ```
    """

    is_train = optimizer is not None
    if is_train:
        model.train()
        torch.set_grad_enabled(True)
    else:
        model.eval()
        torch.set_grad_enabled(False)

    loss_sum = 0.0
    correct = 0
    total = 0

    all_y_true = []
    all_y_prob = []  # probabilities or logits for AUC

    for images, labels in tqdm(loader, disable=not is_train):
        images = images.to(device)
        labels = labels.to(device).squeeze().long()  # [B,1] -> [B]

        if is_train:
            optimizer.zero_grad()

        outputs = model(images)

        if criterion is not None:
            loss = criterion(outputs, labels)
            if is_train:
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * images.size(0)

        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if collect_probs:
            # store per-sample information for AUC etc.
            all_y_true.append(labels.detach().cpu().numpy())
            # use softmax probabilities for multi-class AUC
            probs = torch.softmax(outputs, dim=1).detach().cpu().numpy()
            all_y_prob.append(probs)

    if criterion is not None and len(loader.dataset) > 0:
        loss_avg = loss_sum / len(loader.dataset)
    else:
        # old evaluate() didn't compute loss; just set 0 when criterion is None
        loss_avg = 0.0

    torch.set_grad_enabled(True)

    if collect_probs and all_y_true:
        y_true = np.concatenate(all_y_true, axis=0)
        y_prob = np.concatenate(all_y_prob, axis=0)
    else:
        y_true = None
        y_prob = None

    return {
        "loss_sum": loss_sum,
        "loss_avg": loss_avg,
        "correct": correct,
        "total": total,
        "y_true": y_true,
        "y_prob": y_prob,
    }

def _evaluate_model_core(
    model,
    loader,
    criterion,
    num_classes,
    device=device,
    compute_auc=True,
):
    """
    Core evaluation function that matches your old evaluate_model semantics
    when compute_auc=True:
      - same loss / overall_acc
      - per_class_acc uses np.nan for classes with no samples
      - macro_auc and per_class_auc computed via roc_auc_score with
        multi_class='ovr', average='macro' / None
    """

    # We always want probs for your evaluate_model implementation.
    stats = _run_epoch_core(
        model,
        loader,
        criterion=criterion,
        optimizer=None,
        device=device,
        collect_probs=compute_auc,
    )

    avg_loss = stats["loss_avg"]
    overall_acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0

    per_class_acc = None
    per_class_auc = None
    macro_auc = None

    if num_classes is not None and stats["y_true"] is not None:
        all_labels = stats["y_true"]            # shape [N]
        all_probs = stats["y_prob"]             # shape [N, num_classes]
        all_preds = all_probs.argmax(axis=1)    # [N]

        # ---------- per-class accuracy (same as old) ----------
        per_class_acc = np.zeros(num_classes, dtype=np.float64)
        for c in range(num_classes):
            mask = (all_labels == c)
            if mask.sum() == 0:
                per_class_acc[c] = np.nan  # no samples of this class
            else:
                per_class_acc[c] = (all_preds[mask] == c).mean()

        if compute_auc:
            # ---------- AUCs (same as old) ----------
            macro_auc = roc_auc_score(
                all_labels,
                all_probs,
                multi_class="ovr",
                average="macro",
            )

            per_class_auc = roc_auc_score(
                all_labels,
                all_probs,
                multi_class="ovr",
                average=None,
            )  # shape: [num_classes]

    return {
        "loss": float(avg_loss),
        "overall_acc": float(overall_acc),
        "per_class_acc": per_class_acc,
        "per_class_auc": per_class_auc,
        "macro_auc": float(macro_auc) if macro_auc is not None else None,
    }

def _train_model_core(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    num_epochs,
    num_classes=NUM_CLASSES,
    device=device,
):
    """
    Core training loop.

    :param model: The model to train
    :param train_loader: The dataloader for the training dataset
    :param val_loader: The data loader for validation dataset
    :param optimizer: The optimizer to use.
    :param criterion: The loss function to use.
    :param num_epochs: The number of epochs to train for
    :param num_classes: The number of output classes.
    :param device: The device to train on.
    :returns: model, train_history

    Where train_history has:
        {
          "train_loss": [],
          "train_acc": [],
          "val_loss": [],
          "val_acc": [],
          "val_macro_auc": [],
          "val_per_class_acc": [],
          "val_per_class_auc": [],
        }
    """

    train_history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_auc": [],
        "val_per_class_acc": [],
        "val_per_class_auc": [],
    }

    for epoch in range(num_epochs):
        # ---- training epoch ----
        train_stats = _run_epoch_core(
            model,
            train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            collect_probs=False,
        )
        train_loss = train_stats["loss_avg"]
        train_acc = (
            train_stats["correct"] / train_stats["total"]
            if train_stats["total"] > 0
            else 0.0
        )

        # ---- validation epoch ----
        val_metrics = _evaluate_model_core(
            model,
            val_loader,
            criterion=criterion,
            num_classes=num_classes,
            device=device,
            compute_auc=True,
        )

        train_history["train_loss"].append(train_loss)
        train_history["train_acc"].append(train_acc)

        train_history["val_loss"].append(val_metrics["loss"])
        train_history["val_acc"].append(val_metrics["overall_acc"])
        train_history["val_macro_auc"].append(val_metrics["macro_auc"])
        train_history["val_per_class_acc"].append(val_metrics["per_class_acc"])
        train_history["val_per_class_auc"].append(val_metrics["per_class_auc"])

        macro_auc = val_metrics['macro_auc']
        if macro_auc is None:
            macro_auc_str = "None"
        else:
            macro_auc_str = f"{macro_auc:.3f}"

        print(
            f"Val loss: {val_metrics['loss']:.4f}, "
            f"acc: {val_metrics['overall_acc']:.3f}, "
            f"macro AUC: {macro_auc_str}",
        )

    return model, train_history


# =========================
# WRAPPERS for existing APIs
# =========================

# ---- 1) Wrapper matching the FIRST file's train_model signature ----
#     def train_model(model, train_loader, val_loader, optimizer, criterion,
#                     num_epochs=NUM_EPOCHS, num_classes=NUM_CLASSES, device=device)
#     -> returns train_history (unchanged)

# AI Usage: This function was originally written by me, but was modified by ChatGPT to
# add the metrics per-class accuracy and per-class AUC, as well as macro AUC for each epoch
# AI was later used to refactor this function into the way it is now.
def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    num_epochs,
    num_classes=NUM_CLASSES,
    device=device,
):
    """
    Backwards-compatible with your first train_model:
    - same signature
    - still returns train_history
    """
    _, train_history = _train_model_core(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=num_epochs,
        num_classes=num_classes,
        device=device,
    )
    return train_history


# ---- 2) Wrapper matching the SECOND file's evaluate signature ----
#     def evaluate(model, loader, device=device) -> accuracy float

def evaluate(model, loader, device=device):
    """
    Backwards-compatible with the simpler evaluate:
    - same signature
    - returns only overall accuracy
    - internally uses the richer core evaluation
    """
    metrics = _evaluate_model_core(
        model=model,
        loader=loader,
        criterion=None,
        num_classes=NUM_CLASSES,
        device=device,
        compute_auc=False,  # no AUC, just loss/acc
    )
    return metrics["overall_acc"]

# ---- 3) Wrapper matching the SECOND file's train_model signature ----
#     def train_model(model, train_loader, val_loader, epochs=5,
#                     lr=1e-3, device=device, criterion=nn.CrossEntropyLoss())
#     -> returns model

# AI Usage: asked ChatGPT how to incorperate the learning rate into this function
def train_model_simple(
    model,
    train_loader,
    val_loader,
    epochs=5,
    lr=1e-3,
    device=device,
    criterion=None,
    num_classes=NUM_CLASSES,
):
    """
    Backwards-compatible replacement for the second train_model:
    - nearly identical signature (extra optional num_classes for AUC; can be omitted)
    - returns model
    - loss printed is still epoch-aggregated information
    """

    if criterion is None:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Use the same core training function
    model, history = _train_model_core(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        num_epochs=epochs,
        num_classes=num_classes,
        device=device,
    )

    # If you want to mimic the *exact* print format of the old simple
    # train_model, you can reprint here using history["train_loss"]:
    for epoch in range(epochs):
        # Old simple function printed sum of batch losses.
        # Here we have avg loss; we could approximate sum if needed:
        train_loss_avg = history["train_loss"][epoch]
        n_train = len(train_loader.dataset)
        train_loss_sum_approx = train_loss_avg * max(n_train, 1)

        val_acc = history["val_acc"][epoch]
        print(
            f"[Simple API] Epoch {epoch+1}/{epochs} | "
            f"Loss: {train_loss_sum_approx:.4f} | Val Acc: {val_acc:.4f}"
        )

    return model

# AI Usage: This function was generated by ChatGPT to look at per-class accuracy and AUC
def evaluate_model(model, loader, criterion, num_classes, device=device):
    return _evaluate_model_core(
        model=model,
        loader=loader,
        criterion=criterion,
        num_classes=num_classes,
        device=device,
        compute_auc=True,   # matches old behavior
    )
