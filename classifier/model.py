"""
classifier/model.py

EfficientNet-B0 transfer-learning model for 12-class waste classification.

Public API
----------
build_model(num_classes)         → nn.Module  (untrained, head randomised)
load_trained_model(path)         → (nn.Module, class_names)
predict(model, tensor,
        class_names)             → (label, confidence, all_scores_dict)

Design notes
------------
- Uses EfficientNet_B0_Weights.DEFAULT (current torchvision ≥ 0.13 API).
- The backbone (features) is frozen initially; train.py selectively unfreezes
  deeper blocks after warm-up epochs.
- num_classes and class_names are stored inside the checkpoint so that the
  inference application never requires the training dataset to be present.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_model(num_classes: int) -> nn.Module:
    """
    Load EfficientNet-B0 with ImageNet weights and replace the classification
    head with a new Linear layer sized for num_classes.

    The entire backbone (features) is frozen. Only the new head is trainable
    at construction time. train.py may unfreeze deeper layers after warm-up.

    Parameters
    ----------
    num_classes : int
        Number of output classes (derived from the dataset folder count).

    Returns
    -------
    nn.Module
        Model in train() mode with backbone frozen.
    """
    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)

    # Freeze all backbone parameters.
    for param in model.features.parameters():
        param.requires_grad = False

    # EfficientNet-B0 head: model.classifier = Sequential(Dropout, Linear(1280, 1000))
    # Replace the final Linear with one sized for our task.
    in_features = model.classifier[1].in_features   # 1280
    model.classifier[1] = nn.Linear(in_features, num_classes)
    # The new head has requires_grad=True by default.

    return model


def unfreeze_last_n_blocks(model: nn.Module, n: int = 2) -> None:
    """
    Unfreeze the last n feature stages in the EfficientNet-B0 backbone for
    fine-grained transfer learning after warm-up.

    EfficientNet-B0 model.features — exact structure (9 sub-modules, indices 0–8):
      features[0]   : Conv2dNormActivation  — stem conv (3→32, stride 2)
      features[1]   : Sequential([MBConv])                     — stage 1
      features[2]   : Sequential([MBConv, MBConv])             — stage 2
      features[3]   : Sequential([MBConv, MBConv])             — stage 3
      features[4]   : Sequential([MBConv, MBConv, MBConv])     — stage 4
      features[5]   : Sequential([MBConv, MBConv, MBConv])     — stage 5
      features[6]   : Sequential([MBConv, MBConv, MBConv, MBConv]) — stage 6
      features[7]   : Sequential([MBConv])                     — stage 7
      features[8]   : Conv2dNormActivation  — head conv (320→1280)

    Strategy: unfreeze the last n sub-modules from the END of features.
    Default n=2 unfreezes features[7] (final MBConv stage) and
    features[8] (head conv). These are the highest-level semantic layers
    most likely to benefit from task-specific fine-tuning.
    Stages features[0]–features[6-n] remain frozen throughout.

    Parameters
    ----------
    model : nn.Module
    n : int
        Number of trailing features sub-modules to unfreeze (default 2).
        Counted from the end: n=1 → features[8] only; n=2 → features[7,8];
        n=3 → features[6,7,8], etc.
    """
    total = len(model.features)   # 9
    for i, stage in enumerate(model.features):
        if i >= total - n:
            for param in stage.parameters():
                param.requires_grad = True


# ---------------------------------------------------------------------------
# Checkpoint save / load
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: nn.Module,
    class_names: List[str],
    path: str | Path,
    extra_meta: dict | None = None,
) -> None:
    """
    Save model weights + class_names + class_to_idx into a single checkpoint.

    The class_names list and class_to_idx dict are embedded so that
    load_trained_model() can reconstruct the full prediction pipeline without
    the training dataset being present.

    Parameters
    ----------
    model : nn.Module
    class_names : list[str]
        Ordered list of class names matching model output indices.
    path : str | Path
        Destination .pth file (parent directory must exist).
    extra_meta : dict, optional
        Any additional key/value pairs to store (e.g. val_accuracy).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    payload = {
        "state_dict":    model.state_dict(),
        "class_names":   class_names,
        "class_to_idx":  class_to_idx,
        "num_classes":   len(class_names),
    }
    if extra_meta:
        payload.update(extra_meta)

    torch.save(payload, path)
    print(f"[save_checkpoint] Saved -> {path}")


def load_trained_model(
    path: str | Path,
) -> Tuple[nn.Module, List[str]]:
    """
    Load a trained EfficientNet-B0 model from a checkpoint created by
    save_checkpoint().

    Does NOT require the training dataset to be present.

    Parameters
    ----------
    path : str | Path
        Path to the .pth checkpoint file.

    Returns
    -------
    model : nn.Module
        Model in eval() mode, ready for inference.
    class_names : list[str]
        Ordered class names; index i → class_names[i].

    Raises
    ------
    FileNotFoundError
        If the checkpoint file does not exist.
    KeyError
        If the checkpoint is missing required keys (wrong format).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Trained model not found: {path}\n"
            "Run training first:\n"
            "    python -m classifier.train\n"
            "Ensure the Garbage Classification dataset is available at "
            "data/garbage_classification/ before training."
        )

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    for required_key in ("state_dict", "class_names", "num_classes"):
        if required_key not in checkpoint:
            raise KeyError(
                f"Checkpoint at {path} is missing key '{required_key}'. "
                "It may have been created by an older or incompatible version."
            )

    class_names: List[str] = checkpoint["class_names"]
    num_classes: int = checkpoint["num_classes"]

    model = build_model(num_classes)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    return model, class_names


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict(
    model: nn.Module,
    tensor: torch.Tensor,
    class_names: List[str],
) -> Tuple[str, float, Dict[str, float]]:
    """
    Run a single forward pass and return the top prediction.

    Parameters
    ----------
    model : nn.Module
        Loaded model in eval() mode.
    tensor : torch.Tensor
        Preprocessed image tensor of shape (1, 3, 224, 224).
    class_names : list[str]
        Class names in index order (from checkpoint).

    Returns
    -------
    label : str
        Predicted class name.
    confidence : float
        Softmax probability of the top class (0.0 – 1.0).
    all_scores : dict[str, float]
        Softmax probability for every class, keyed by class name.
        Useful for displaying a full score breakdown in the UI.
    """
    model.eval()
    with torch.no_grad():
        logits = model(tensor)                    # (1, num_classes)
        probs  = F.softmax(logits, dim=1)[0]      # (num_classes,)

    top_idx    = int(probs.argmax().item())
    confidence = float(probs[top_idx].item())
    label      = class_names[top_idx]

    all_scores = {
        class_names[i]: float(probs[i].item())
        for i in range(len(class_names))
    }

    return label, confidence, all_scores
