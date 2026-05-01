from __future__ import annotations

from dataclasses import replace
from typing import Any

from PIL import Image

from .config import DATASET_NAME, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD


def pil_from_example(example: dict[str, Any]) -> Image.Image:
    image = example.get("image") or example.get("img")
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    raise TypeError("Expected a PIL image in the dataset example under 'image' or 'img'.")


def label_from_example(example: dict[str, Any]) -> int:
    for key in ("label", "labels", "class_id", "ClassId"):
        if key in example:
            return int(example[key])
    raise KeyError("Could not find a label column in the dataset example.")


class GTSRBTorchDataset:
    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.hf_dataset)

    def __getitem__(self, index: int):
        example = self.hf_dataset[index]
        image = pil_from_example(example)
        label = label_from_example(example)
        if self.transform:
            image = self.transform(image)
        return image, label


def build_transforms(image_size: int = IMAGE_SIZE, train: bool = False, use_randaugment: bool = False):
    from torchvision import transforms

    ops = [transforms.Resize((image_size, image_size))]
    if train:
        ops.extend(
            [
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.08),
            ]
        )
        if use_randaugment:
            ops.append(transforms.RandAugment(num_ops=2, magnitude=9))
    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return transforms.Compose(ops)


def load_hf_gtsrb(split: str):
    from datasets import load_dataset

    return load_dataset(DATASET_NAME, split=split)


def stratified_train_val_split(train_dataset, val_fraction: float, seed: int):
    from sklearn.model_selection import train_test_split

    labels = [label_from_example(train_dataset[i]) for i in range(len(train_dataset))]
    indices = list(range(len(train_dataset)))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=val_fraction,
        random_state=seed,
        stratify=labels,
    )
    return {"train": train_dataset.select(train_idx), "test": train_dataset.select(val_idx)}


def make_dataloaders(config):
    from torch.utils.data import DataLoader

    raw_train = load_hf_gtsrb("train")
    split = stratified_train_val_split(raw_train, config.val_fraction, config.seed)
    train_ds = GTSRBTorchDataset(
        split["train"],
        transform=build_transforms(config.image_size, train=True, use_randaugment=config.use_randaugment),
    )
    val_ds = GTSRBTorchDataset(
        split["test"],
        transform=build_transforms(config.image_size, train=False),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader


def config_with_updates(config, **updates):
    return replace(config, **updates)
