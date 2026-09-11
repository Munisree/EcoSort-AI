"""
classifier/labels.py

Provides a helper to derive the class-name list directly from the dataset
folder structure via torchvision.datasets.ImageFolder.

IMPORTANT: Class names are NEVER hard-coded here. The index→label mapping is
always derived at runtime from the actual folders that exist on disk during
training, and then persisted inside the model checkpoint so that inference
does not require the dataset to be present.
"""

import os
from pathlib import Path

from torchvision.datasets import ImageFolder


def get_class_names(dataset_root: str | Path) -> list[str]:
    """
    Read class names from the ImageFolder directory structure.

    Parameters
    ----------
    dataset_root : str | Path
        Root directory whose immediate subdirectories are the class folders,
        e.g. data/garbage_classification/  ← contains battery/, paper/, …

    Returns
    -------
    list[str]
        Class names in index order as assigned by ImageFolder (alphabetical).
        Index 0 → class_names[0], etc.

    Raises
    ------
    FileNotFoundError
        If dataset_root does not exist or contains no subdirectories.
    """
    dataset_root = Path(dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(
            f"Dataset root not found: {dataset_root}\n"
            "Download the Garbage Classification dataset and place it at that path."
        )

    # ImageFolder.classes is already sorted alphabetically — same ordering
    # torchvision uses when assigning integer targets during training.
    dataset = ImageFolder(root=str(dataset_root))
    if not dataset.classes:
        raise FileNotFoundError(
            f"No class subdirectories found in: {dataset_root}"
        )
    return dataset.classes
