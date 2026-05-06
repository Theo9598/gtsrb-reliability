from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from .config import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD
from .gradcam import GradCAM, find_last_conv_layer
from .labels import GTSRB_LABELS, NUM_CLASSES
from .models import build_model


def load_checkpoint_state(checkpoint: Path, device=None):
    import torch

    map_location = device if device is not None else "cpu"
    try:
        return torch.load(checkpoint, map_location=map_location)
    except Exception:
        return torch.load(checkpoint, map_location=map_location, weights_only=False)


def infer_model_name_from_checkpoint(checkpoint: Path, default: str = "resnet18") -> str:
    state = load_checkpoint_state(checkpoint)
    config = state.get("config", {}) if isinstance(state, dict) else {}
    return str(config.get("model") or default)


def load_model_for_inference(checkpoint: Path, model_name: str | None = None):
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state = load_checkpoint_state(checkpoint, device=device)
    resolved_model = model_name or str(state.get("config", {}).get("model") or "resnet18")
    model = build_model(resolved_model, NUM_CLASSES, pretrained=False).to(device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model, device, resolved_model


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
