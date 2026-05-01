from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

from .augment import mixup_batch, mixup_loss
from .config import TrainConfig
from .data import config_with_updates, make_dataloaders
from .models import build_model, set_backbone_trainable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a GTSRB classifier.")
    parser.add_argument(
        "--model",
        default="resnet18",
        choices=["baseline_cnn", "baseline_cnn_no_bn", "baseline_cnn_no_dropout", "resnet18", "efficientnet_b0"],
    )
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--mixup-alpha", type=float, default=0.0)
    parser.add_argument("--use-randaugment", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/resnet18_robust"))
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=242)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def serializable_config(config: TrainConfig) -> dict:
    values = asdict(config)
    values["output_dir"] = str(values["output_dir"])
    return values


def run_epoch(model, loader, criterion, optimizer, device, mixup_alpha: float = 0.0) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    correct = 0
    total = 0
    iterator = tqdm(loader, leave=False)
    for images, labels in iterator:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
            if mixup_alpha > 0:
                images, labels_a, labels_b, lam = mixup_batch(images, labels, mixup_alpha)
                logits = model(images)
                loss = mixup_loss(criterion, logits, labels_a, labels_b, lam)
            else:
                logits = model(images)
                loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        else:
            with torch.no_grad():
                logits = model(images)
                loss = criterion(logits, labels)
        total_loss += float(loss.item()) * images.size(0)
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        total += int(images.size(0))
        iterator.set_postfix(loss=total_loss / max(total, 1), acc=correct / max(total, 1))
    return {"loss": total_loss / total, "accuracy": correct / total}


def train(config: TrainConfig) -> Path:
    seed_everything(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = make_dataloaders(config)
    model = build_model(config.model, config.num_classes, pretrained=config.model != "baseline_cnn").to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    history = []
    best_val_acc = -1.0
    for epoch in range(1, config.epochs + 1):
        if config.model != "baseline_cnn" and epoch == 1:
            set_backbone_trainable(model, trainable=False)
        if config.model != "baseline_cnn" and epoch == config.freeze_backbone_epochs + 1:
            set_backbone_trainable(model, trainable=True)
            optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate / 3, weight_decay=config.weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, config.epochs - epoch + 1))

        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, config.mixup_alpha)
        val_metrics = run_epoch(model, val_loader, criterion, None, device)
        scheduler.step()
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "learning_rate": scheduler.get_last_lr()[0],
        }
        history.append(row)
        pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": serializable_config(config),
                    "best_val_accuracy": best_val_acc,
                },
                output_dir / "best.pt",
            )
        print(json.dumps(row, indent=2))

    (output_dir / "config.json").write_text(json.dumps(serializable_config(config), indent=2))
    return output_dir / "best.pt"


def main() -> None:
    args = parse_args()
    config = config_with_updates(
        TrainConfig(),
        model=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        mixup_alpha=args.mixup_alpha,
        use_randaugment=args.use_randaugment,
        output_dir=args.output_dir,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    train(config)


if __name__ == "__main__":
    main()
