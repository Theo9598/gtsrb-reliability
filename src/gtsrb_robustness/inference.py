from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD
from .gradcam import GradCAM, find_last_conv_layer
from .labels import GTSRB_LABELS, NUM_CLASSES
from .models import build_model


def load_model_for_inference(checkpoint: Path, model_name: str = "resnet18"):
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_name, NUM_CLASSES, pretrained=False).to(device)
    try:
        state = torch.load(checkpoint, map_location=device)
    except Exception:
        state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model, device


def preprocess_image(image: Image.Image):
    from torchvision import transforms

    transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return transform(image.convert("RGB")).unsqueeze(0)


def predict_topk(model, device, image: Image.Image, k: int = 5, temperature: float = 1.0) -> list[dict]:
    import torch

    tensor = preprocess_image(image).to(device)
    with torch.no_grad():
        logits = model(tensor) / max(float(temperature), 1e-4)
        probs = torch.softmax(logits, dim=1)[0]
        values, indices = torch.topk(probs, k=k)
    return [
        {"class_id": int(idx), "label": GTSRB_LABELS[int(idx)], "probability": float(prob)}
        for idx, prob in zip(indices.cpu(), values.cpu())
    ]


def gradcam_overlay(model, device, image: Image.Image, alpha: float = 0.42) -> Image.Image:
    import matplotlib.cm as cm

    tensor = preprocess_image(image).to(device)
    cam = GradCAM(model, find_last_conv_layer(model))
    try:
        heatmap = cam(tensor)
    finally:
        cam.remove_hooks()
    base = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    heat_rgba = cm.get_cmap("turbo")(heatmap)
    heat_rgb = (heat_rgba[:, :, :3] * 255).astype(np.uint8)
    blended = (np.asarray(base).astype(np.float32) * (1 - alpha) + heat_rgb.astype(np.float32) * alpha)
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))
