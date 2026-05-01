from __future__ import annotations

import torch


def mixup_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if alpha <= 0:
        return images, labels, labels, 1.0
    lam = torch.distributions.Beta(alpha, alpha).sample().item()
    index = torch.randperm(images.size(0), device=images.device)
    mixed_images = lam * images + (1.0 - lam) * images[index]
    labels_a = labels
    labels_b = labels[index]
    return mixed_images, labels_a, labels_b, float(lam)


def mixup_loss(criterion, logits, labels_a, labels_b, lam: float):
    return lam * criterion(logits, labels_a) + (1.0 - lam) * criterion(logits, labels_b)

