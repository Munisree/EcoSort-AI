"""
classifier/train.py

EfficientNet-B0 transfer-learning training script for 12-class waste
classification on the Garbage Classification dataset.

Usage
-----
    python -m classifier.train

    Optional arguments (environment variables or edit CONFIG below):
        ECOSORT_DATA_DIR   path to dataset root  (default: data/garbage_classification)
        ECOSORT_MODEL_PATH path to save .pth      (default: models/ecosort_efficientnet_b0.pth)
        ECOSORT_EPOCHS     total training epochs   (default: 5)
        ECOSORT_BATCH      batch size              (default: 16)
        ECOSORT_LR         initial learning rate   (default: 1e-3)
        ECOSORT_WARMUP     warm-up epochs          (default: 2)
        ECOSORT_UNFREEZE   feature stages to unfreeze (default: 2)

Dataset layout expected
-----------------------
    data/garbage_classification/
        battery/
            img1.jpg ...
        biological/
            img1.jpg ...
        ... (one subfolder per class)

Training procedure
------------------
1. Read class names from ImageFolder (alphabetical → consistent index order).
2. Stratified 80 / 10 / 10 split (train / val / test) using per-class indices
   so that each class is proportionally represented in every split.
3. Warm-up phase (epochs 1 – WARMUP): backbone frozen, train head only.
4. Fine-tune phase (epoch WARMUP+1 – END): last UNFREEZE feature stages
   unfrozen (features[9-UNFREEZE .. 8]), lower LR applied to those params.
   Default UNFREEZE=2 unfreezes features[7] (final MBConv stage) and
   features[8] (head Conv2dNormActivation). See model.py for full structure.
5. Best checkpoint saved whenever validation loss improves.
6. Final evaluation on held-out test set: accuracy + classification_report.

Checkpoint format (saved to MODEL_PATH)
----------------------------------------
{
    "state_dict":   <OrderedDict>,
    "class_names":  ["battery", "biological", ...],   # alphabetical
    "class_to_idx": {"battery": 0, "biological": 1, ...},
    "num_classes":  12,
    "val_accuracy": <float>,
    "test_accuracy": <float>,
    "epochs_trained": <int>,
}
"""

from __future__ import annotations

import os
import time
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedShuffleSplit

from classifier.labels import get_class_names
from classifier.model import (
    build_model,
    save_checkpoint,
    unfreeze_last_n_blocks,
)
from classifier.preprocess import training_transform, val_transform

# ---------------------------------------------------------------------------
# Configuration — override via environment variables or edit here.
# ---------------------------------------------------------------------------
CONFIG = {
    "data_dir":    os.environ.get("ECOSORT_DATA_DIR",   "data/garbage_classification"),
    "model_path":  os.environ.get("ECOSORT_MODEL_PATH", "models/ecosort_efficientnet_b0.pth"),
    "epochs":      int(os.environ.get("ECOSORT_EPOCHS",   "5")),
    "batch_size":  int(os.environ.get("ECOSORT_BATCH",    "16")),
    "lr":          float(os.environ.get("ECOSORT_LR",     "1e-3")),
    "warmup":      int(os.environ.get("ECOSORT_WARMUP",   "2")),
    "unfreeze_n":  int(os.environ.get("ECOSORT_UNFREEZE", "2")),
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Stratified split
# ---------------------------------------------------------------------------

def stratified_split(
    dataset: ImageFolder,
    val_ratio: float = 0.10,
    test_ratio: float = 0.10,
    random_state: int = 42,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Return (train_indices, val_indices, test_indices) using a two-stage
    StratifiedShuffleSplit so that each class is proportionally represented
    in every split.

    Stage 1: split full dataset → (train+val) / test
    Stage 2: split (train+val)  → train / val
    """
    targets = np.array(dataset.targets)
    indices = np.arange(len(targets))

    # Stage 1: carve out test set.
    sss1 = StratifiedShuffleSplit(
        n_splits=1, test_size=test_ratio, random_state=random_state
    )
    trainval_idx, test_idx = next(sss1.split(indices, targets))

    # Stage 2: split remainder into train / val.
    val_relative = val_ratio / (1.0 - test_ratio)   # adjust ratio for the subset
    sss2 = StratifiedShuffleSplit(
        n_splits=1, test_size=val_relative, random_state=random_state
    )
    train_idx_rel, val_idx_rel = next(
        sss2.split(trainval_idx, targets[trainval_idx])
    )

    train_idx = trainval_idx[train_idx_rel]
    val_idx   = trainval_idx[val_idx_rel]

    return train_idx.tolist(), val_idx.tolist(), test_idx.tolist()


def _verify_split_class_balance(
    dataset: ImageFolder,
    train_idx: List[int],
    val_idx: List[int],
    test_idx: List[int],
) -> None:
    """Print per-split class counts to verify stratification."""
    targets = np.array(dataset.targets)

    def counts(idx_list):
        c = defaultdict(int)
        for i in idx_list:
            c[dataset.classes[targets[i]]] += 1
        return c

    print("\n[split] Class distribution verification:")
    print(f"  {'Class':<20} {'Train':>6} {'Val':>6} {'Test':>6}")
    print(f"  {'-'*44}")
    for cls in dataset.classes:
        tr = counts(train_idx).get(cls, 0)
        va = counts(val_idx).get(cls, 0)
        te = counts(test_idx).get(cls, 0)
        print(f"  {cls:<20} {tr:>6} {va:>6} {te:>6}")
    print(
        f"  {'TOTAL':<20} {len(train_idx):>6} {len(val_idx):>6} {len(test_idx):>6}\n"
    )


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _make_loaders(
    data_dir: str,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """Build train / val / test DataLoaders with stratified splits."""

    # Load full dataset with the training transform first (we will override
    # for val/test subsets below).
    full_dataset = ImageFolder(root=data_dir, transform=training_transform)
    class_names  = full_dataset.classes

    print(f"[data] Found {len(full_dataset)} images in {len(class_names)} classes.")
    print(f"[data] Classes: {class_names}")

    train_idx, val_idx, test_idx = stratified_split(full_dataset)
    _verify_split_class_balance(full_dataset, train_idx, val_idx, test_idx)

    # Val and test use the deterministic val_transform, not training augmentation.
    # We achieve this by creating separate ImageFolder instances with different
    # transforms and then indexing with the same indices.
    val_dataset  = ImageFolder(root=data_dir, transform=val_transform)
    test_dataset = ImageFolder(root=data_dir, transform=val_transform)

    train_loader = DataLoader(
        Subset(full_dataset, train_idx),
        batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True,
    )
    val_loader   = DataLoader(
        Subset(val_dataset,  val_idx),
        batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True,
    )
    test_loader  = DataLoader(
        Subset(test_dataset, test_idx),
        batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True,
    )

    return train_loader, val_loader, test_loader, class_names


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    phase: str,
) -> Tuple[float, float]:
    """
    Run one training or evaluation epoch.

    Returns
    -------
    avg_loss : float
    accuracy : float  (0.0 – 1.0)
    """
    if phase == "train":
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    correct      = 0
    total        = 0

    context = torch.enable_grad() if phase == "train" else torch.no_grad()

    with context:
        for images, labels in loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            if phase == "train":
                optimizer.zero_grad()

            outputs = model(images)
            loss    = criterion(outputs, labels)

            if phase == "train":
                loss.backward()
                optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds         = outputs.argmax(dim=1)
            correct      += (preds == labels).sum().item()
            total        += images.size(0)

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


# ---------------------------------------------------------------------------
# Evaluation on test set
# ---------------------------------------------------------------------------

def evaluate_test_set(
    model: nn.Module,
    test_loader: DataLoader,
    class_names: List[str],
) -> float:
    """
    Run inference on the held-out test set and print:
    - Overall accuracy
    - sklearn classification_report (precision / recall / F1 per class)

    Returns
    -------
    test_accuracy : float
    """
    model.eval()
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images  = images.to(DEVICE)
            outputs = model(images)
            preds   = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    test_accuracy = np.mean(np.array(all_preds) == np.array(all_labels))

    print("\n" + "=" * 60)
    print(f"TEST SET RESULTS")
    print("=" * 60)
    print(f"  Overall accuracy: {test_accuracy * 100:.2f}%")
    print()
    print(
        classification_report(
            all_labels, all_preds, target_names=class_names, digits=3
        )
    )
    print("=" * 60 + "\n")

    return float(test_accuracy)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train() -> None:
    cfg = CONFIG
    print(f"\n{'='*60}")
    print(f"EcoSort AI — Phase 1 Training")
    print(f"{'='*60}")
    print(f"  Device:     {DEVICE}")
    print(f"  Data dir:   {cfg['data_dir']}")
    print(f"  Model path: {cfg['model_path']}")
    print(f"  Epochs:     {cfg['epochs']}  (warm-up: {cfg['warmup']})")
    print(f"  Batch size: {cfg['batch_size']}")
    print(f"  LR:         {cfg['lr']}")
    print(f"  Unfreeze:   last {cfg['unfreeze_n']} feature stage(s) after warm-up")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------ Data
    train_loader, val_loader, test_loader, class_names = _make_loaders(
        cfg["data_dir"], cfg["batch_size"]
    )
    num_classes = len(class_names)

    # ----------------------------------------------------------------- Model
    model = build_model(num_classes).to(DEVICE)
    print(f"[model] EfficientNet-B0 loaded. Head: Linear(1280, {num_classes})")
    print(f"[model] Backbone frozen for warm-up phase ({cfg['warmup']} epochs).")

    criterion = nn.CrossEntropyLoss()

    # Only the classifier head is trainable during warm-up.
    head_params    = [p for p in model.classifier.parameters() if p.requires_grad]
    optimizer      = torch.optim.Adam(head_params, lr=cfg["lr"])
    scheduler      = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
        # verbose= was removed in PyTorch 2.x and raises TypeError if passed.
        # LR changes are logged explicitly in the epoch loop below instead.
    )

    best_val_loss   = float("inf")
    best_val_acc    = 0.0
    epochs_trained  = 0
    unfrozen        = False

    # --------------------------------------------------------------- Training
    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()

        # After warm-up: unfreeze last N feature stages, add to optimiser.
        if epoch == cfg["warmup"] + 1 and not unfrozen:
            print(
                f"\n[train] Warm-up complete. Unfreezing last "
                f"{cfg['unfreeze_n']} feature stage(s) "
                f"(features[{9 - cfg['unfreeze_n']}..8]).\n"
            )
            unfreeze_last_n_blocks(model, cfg["unfreeze_n"])
            # Add newly unfrozen params at a lower LR to avoid destroying
            # pre-trained features.
            backbone_params = [
                p for p in model.features.parameters() if p.requires_grad
            ]
            optimizer.add_param_group({"params": backbone_params, "lr": cfg["lr"] * 0.1})
            unfrozen = True

        train_loss, train_acc = _run_epoch(model, train_loader, criterion, optimizer, "train")
        val_loss,   val_acc   = _run_epoch(model, val_loader,   criterion, None,      "val")

        scheduler.step(val_loss)

        elapsed = time.time() - t0
        print(
            f"Epoch [{epoch:>3}/{cfg['epochs']}]  "
            f"train loss: {train_loss:.4f}  train acc: {train_acc*100:>6.2f}%  |  "
            f"val loss: {val_loss:.4f}  val acc: {val_acc*100:>6.2f}%  "
            f"({elapsed:.1f}s)"
        )

        # Save checkpoint whenever validation loss improves.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc  = val_acc
            epochs_trained = epoch
            save_checkpoint(
                model,
                class_names,
                cfg["model_path"],
                extra_meta={
                    "val_accuracy":    best_val_acc,
                    "epochs_trained":  epoch,
                },
            )
            print(f"  [saved] Best checkpoint saved (val loss: {best_val_loss:.4f})")

    # ------------------------------------------------------- Test evaluation
    print(f"\n[eval] Loading best checkpoint from {cfg['model_path']} for test evaluation ...")
    best_model, _ = _load_best_for_eval(cfg["model_path"], num_classes)
    test_accuracy  = evaluate_test_set(best_model, test_loader, class_names)

    # Update checkpoint with test accuracy.
    checkpoint = torch.load(cfg["model_path"], map_location="cpu", weights_only=False)
    checkpoint["test_accuracy"] = test_accuracy
    torch.save(checkpoint, cfg["model_path"])
    print(f"[eval] test_accuracy written to checkpoint: {test_accuracy*100:.2f}%")


def _load_best_for_eval(path: str, num_classes: int) -> tuple:
    """Load the best saved checkpoint for final test evaluation."""
    from classifier.model import load_trained_model
    return load_trained_model(path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train()
