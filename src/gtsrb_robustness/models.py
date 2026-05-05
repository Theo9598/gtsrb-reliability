from __future__ import annotations

import torch
from torch import nn

from .labels import NUM_CLASSES


class BaselineCNN(nn.Module):
    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        use_batchnorm: bool = True,
        dropout_p: float = 0.35,
    ):
        super().__init__()
        def norm(channels: int) -> nn.Module:
            return nn.BatchNorm2d(channels) if use_batchnorm else nn.Identity()

        classifier_layers: list[nn.Module] = [nn.Flatten()]
        if dropout_p > 0:
            classifier_layers.append(nn.Dropout(dropout_p))
        classifier_layers.append(nn.Linear(256, num_classes))

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            norm(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            norm(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            norm(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            norm(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def build_model(model_name: str, num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    name = model_name.lower()
    if name == "baseline_cnn":
        return BaselineCNN(num_classes=num_classes)
    if name == "baseline_cnn_no_bn":
        return BaselineCNN(num_classes=num_classes, use_batchnorm=False)
    if name == "baseline_cnn_no_dropout":
        return BaselineCNN(num_classes=num_classes, dropout_p=0.0)

    from torchvision import models

    if name == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    raise ValueError(f"Unknown model: {model_name}")


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for name, param in model.named_parameters():
        is_head = name.startswith("fc.") or name.startswith("classifier.")
        param.requires_grad = trainable or is_head
