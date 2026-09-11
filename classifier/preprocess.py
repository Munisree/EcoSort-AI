"""
classifier/preprocess.py

Single-responsibility module: convert a PIL Image into a normalised
torch.Tensor ready for EfficientNet-B0 inference or training.

The transforms here mirror the preprocessing expected by EfficientNet-B0
trained on ImageNet (EfficientNet_B0_Weights.DEFAULT.transforms()).
They are defined explicitly so that both train.py and app.py share
exactly the same preprocessing pipeline.
"""

import torch
from PIL import Image
from torchvision import transforms

# EfficientNet-B0 input specification
_INPUT_SIZE = 224

# ImageNet channel statistics (mean / std per channel)
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------
# Transform pipelines
# ---------------------------------------------------------------------------

# Inference transform: deterministic, no augmentation.
inference_transform = transforms.Compose([
    transforms.Resize((_INPUT_SIZE, _INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])

# Training transform: includes light augmentation to improve generalisation.
training_transform = transforms.Compose([
    transforms.Resize((_INPUT_SIZE + 32, _INPUT_SIZE + 32)),   # slight oversize
    transforms.RandomCrop(_INPUT_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])

# Validation / test transform: same as inference (deterministic).
val_transform = inference_transform


def preprocess_image(pil_image: Image.Image) -> torch.Tensor:
    """
    Preprocess a single PIL Image for inference.

    Parameters
    ----------
    pil_image : PIL.Image.Image
        Raw image loaded from an uploaded file or path. EXIF orientation is
        NOT applied automatically here; callers should open with
        ImageOps.exif_transpose() if needed.

    Returns
    -------
    torch.Tensor
        Shape: (1, 3, 224, 224) — batch dimension included, ready for
        model(tensor) without an additional unsqueeze call.
    """
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")

    tensor = inference_transform(pil_image)   # (3, 224, 224)
    return tensor.unsqueeze(0)                # (1, 3, 224, 224)
