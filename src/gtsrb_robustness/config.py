from dataclasses import dataclass
from pathlib import Path

from .labels import NUM_CLASSES

DATASET_NAME = "tanganke/gtsrb"

CORRUPTED_SPLITS = [
    "contrast",
    "gaussian_noise",
    "impulse_noise",
    "jpeg_compression",
    "motion_blur",
    "pixelate",
    "spatter",
]

IMAGE_SIZE = 96
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class TrainConfig:
    model: str = "resnet18"
    image_size: int = IMAGE_SIZE
    num_classes: int = NUM_CLASSES
    epochs: int = 25
    batch_size: int = 96
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    val_fraction: float = 0.15
    num_workers: int = 4
    seed: int = 242
    mixup_alpha: float = 0.0
    use_randaugment: bool = False
    freeze_backbone_epochs: int = 2
    output_dir: Path = Path("runs/resnet18_robust")

