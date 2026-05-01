from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from .calibration import fit_temperature
from .config import CORRUPTED_SPLITS, IMAGE_SIZE
from .corruptions import EVAL_CORRUPTIONS
from .data import GTSRBTorchDataset, build_transforms, load_hf_gtsrb, stratified_train_val_split
from .data import label_from_example, pil_from_example
from .labels import GTSRB_LABELS, NUM_CLASSES
from .metrics import expected_calibration_error, reliability_bins, softmax
from .models import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained GTSRB classifier.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--model",
        default="resnet18",
        choices=["baseline_cnn", "baseline_cnn_no_bn", "baseline_cnn_no_dropout", "resnet18", "efficientnet_b0"],
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/evaluation"))
    parser.add_argument("--skip-corruptions", action="store_true")
    return parser.parse_args()


def load_checkpoint_model(checkpoint: Path, model_name: str, device: torch.device):
    model = build_model(model_name, NUM_CLASSES, pretrained=False).to(device)
    try:
        state = torch.load(checkpoint, map_location=device)
    except Exception:
        state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model


def collect_logits_from_dataset(
    model,
    hf_dataset,
    name: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    pil_corruption=None,
):
    base_transform = build_transforms(IMAGE_SIZE, train=False)

    def transform(image):
        if pil_corruption is not None:
            image = pil_corruption(image)
        return base_transform(image)

    dataset = GTSRBTorchDataset(hf_dataset, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    logits_rows = []
    label_rows = []
    with torch.no_grad():
        for images, labels in tqdm(loader, leave=False, desc=name):
            images = images.to(device, non_blocking=True)
            logits = model(images).cpu()
            logits_rows.append(logits)
            label_rows.append(labels)
    return torch.cat(logits_rows), torch.cat(label_rows)


def collect_logits(model, dataset_split: str, batch_size: int, num_workers: int, device: torch.device):
    return collect_logits_from_dataset(
        model,
        load_hf_gtsrb(dataset_split),
        dataset_split,
        batch_size,
        num_workers,
        device,
    )


def summarize_predictions(logits: torch.Tensor, labels: torch.Tensor) -> dict:
    logits_np = logits.numpy()
    labels_np = labels.numpy()
    preds = logits_np.argmax(axis=1)
    probs = softmax(logits_np)
    return {
        "accuracy": float(accuracy_score(labels_np, preds)),
        "macro_f1": float(f1_score(labels_np, preds, average="macro")),
        "ece": expected_calibration_error(probs, labels_np),
        "classification_report": classification_report(
            labels_np,
            preds,
            labels=list(range(NUM_CLASSES)),
            target_names=[GTSRB_LABELS[i] for i in range(NUM_CLASSES)],
            output_dict=True,
            zero_division=0,
        ),
    }


def bootstrap_accuracy_ci(
    logits: torch.Tensor,
    labels: torch.Tensor,
    n_bootstrap: int = 1000,
    seed: int = 242,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    preds = logits.argmax(dim=1).numpy()
    labels_np = labels.numpy()
    n = len(labels_np)
    scores = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        scores.append(float(np.mean(preds[idx] == labels_np[idx])))
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def plot_confusion(logits: torch.Tensor, labels: torch.Tensor, output_path: Path) -> None:
    cm = confusion_matrix(labels.numpy(), logits.argmax(dim=1).numpy(), labels=list(range(NUM_CLASSES)))
    plt.figure(figsize=(8, 7))
    tick_labels = [str(i) if i % 2 == 0 else "" for i in range(NUM_CLASSES)]
    sns.heatmap(
        cm,
        cmap="mako",
        cbar=True,
        square=True,
        xticklabels=tick_labels,
        yticklabels=tick_labels,
        cbar_kws={"label": "Count", "shrink": 0.75},
    )
    plt.title("GTSRB confusion matrix")
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_reliability(probs: np.ndarray, labels: np.ndarray, output_path: Path) -> None:
    rows = reliability_bins(probs, labels, n_bins=15)
    frame = pd.DataFrame(rows)
    centers = (frame["bin_lower"] + frame["bin_upper"]) / 2
    plt.figure(figsize=(7, 6))
    plt.plot([0, 1], [0, 1], linestyle="--", color="#667085", label="Perfect calibration")
    plt.bar(centers, frame["accuracy"], width=1 / 15 * 0.85, color="#2f6fed", alpha=0.72, label="Empirical accuracy")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.02)
    plt.title("Reliability diagram")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_robustness(rows: list[dict], output_path: Path) -> None:
    frame = pd.DataFrame(rows)
    frame["display_split"] = frame.apply(
        lambda row: f"{row['split']}\n({row['source'].replace('_', ' ')})",
        axis=1,
    )
    frame["accuracy_pct"] = frame["accuracy"] * 100
    plt.figure(figsize=(11, 5.4))
    ax = sns.barplot(data=frame, x="display_split", y="accuracy_pct", color="#2f6fed")
    if {"accuracy_ci_low", "accuracy_ci_high"}.issubset(frame.columns):
        for idx, row in frame.reset_index(drop=True).iterrows():
            ax.errorbar(
                idx,
                row["accuracy_pct"],
                yerr=[
                    [(row["accuracy"] - row["accuracy_ci_low"]) * 100],
                    [(row["accuracy_ci_high"] - row["accuracy"]) * 100],
                ],
                color="#101828",
                capsize=3,
                linewidth=1,
            )
    for idx, row in frame.reset_index(drop=True).iterrows():
        ax.text(idx, min(row["accuracy_pct"] + 2.0, 102), f"{row['accuracy_pct']:.1f}", ha="center", fontsize=7)
    plt.xticks(rotation=35, ha="right", fontsize=8)
    plt.ylim(0, 105)
    plt.xlabel("Evaluation split")
    plt.ylabel("Accuracy (%)")
    plt.title("Robustness across clean and corrupted GTSRB splits")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()


def plot_error_examples(model, device: torch.device, output_path: Path, max_examples: int = 8) -> None:
    dataset = load_hf_gtsrb("test")
    transform = build_transforms(IMAGE_SIZE, train=False)
    examples = []
    with torch.no_grad():
        for i in range(len(dataset)):
            image = pil_from_example(dataset[i])
            label = label_from_example(dataset[i])
            tensor = transform(image).unsqueeze(0).to(device)
            pred = int(model(tensor).argmax(dim=1).item())
            if pred != label:
                examples.append((image, label, pred))
            if len(examples) >= max_examples:
                break
    if not examples:
        return
    cols = 4
    rows = int(np.ceil(len(examples) / cols))
    plt.figure(figsize=(12, 3.1 * rows))
    for idx, (image, label, pred) in enumerate(examples):
        ax = plt.subplot(rows, cols, idx + 1)
        ax.imshow(image)
        ax.set_title(f"true {label}: {GTSRB_LABELS[label]}\npred {pred}: {GTSRB_LABELS[pred]}", fontsize=7)
        ax.axis("off")
    plt.suptitle("Representative clean-test errors", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(output_path, dpi=220)
    plt.close()


def evaluate(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint_model(args.checkpoint, args.model, device)

    clean_logits, clean_labels = collect_logits(model, "test", args.batch_size, args.num_workers, device)
    clean_summary = summarize_predictions(clean_logits, clean_labels)
    clean_ci = bootstrap_accuracy_ci(clean_logits, clean_labels)
    plot_confusion(clean_logits, clean_labels, args.output_dir / "confusion_matrix.png")
    plot_error_examples(model, device, args.output_dir / "error_examples.png")

    train_split = stratified_train_val_split(load_hf_gtsrb("train"), val_fraction=0.15, seed=242)
    val_logits, val_labels = collect_logits_from_dataset(
        model,
        train_split["test"],
        "calibration_val",
        args.batch_size,
        args.num_workers,
        device,
    )
    scaler = fit_temperature(val_logits.to(device), val_labels.to(device))
    calibrated_logits = scaler(clean_logits.to(device)).detach().cpu()
    calibrated_probs = softmax(calibrated_logits.numpy())
    calibrated_summary = summarize_predictions(calibrated_logits, clean_labels)
    plot_reliability(calibrated_probs, clean_labels.numpy(), args.output_dir / "reliability_diagram.png")

    split_rows = [
        {
            "split": "test",
            "source": "clean",
            "accuracy": clean_summary["accuracy"],
            "accuracy_ci_low": clean_ci[0],
            "accuracy_ci_high": clean_ci[1],
            "macro_f1": clean_summary["macro_f1"],
        }
    ]
    if not args.skip_corruptions:
        clean_test = load_hf_gtsrb("test")
        for corruption_name, corruption_fn in EVAL_CORRUPTIONS.items():
            logits, labels = collect_logits_from_dataset(
                model,
                clean_test,
                corruption_name,
                args.batch_size,
                args.num_workers,
                device,
                pil_corruption=corruption_fn,
            )
            summary = summarize_predictions(logits, labels)
            ci_low, ci_high = bootstrap_accuracy_ci(logits, labels)
            split_rows.append(
                {
                    "split": corruption_name,
                    "source": "local_corruption",
                    "accuracy": summary["accuracy"],
                    "accuracy_ci_low": ci_low,
                    "accuracy_ci_high": ci_high,
                    "macro_f1": summary["macro_f1"],
                }
            )
        for split_name in CORRUPTED_SPLITS:
            try:
                logits, labels = collect_logits(model, split_name, args.batch_size, args.num_workers, device)
            except Exception as exc:
                print(f"Skipping split {split_name}: {exc}")
                continue
            summary = summarize_predictions(logits, labels)
            ci_low, ci_high = bootstrap_accuracy_ci(logits, labels)
            split_rows.append(
                {
                    "split": split_name,
                    "source": "hf_split",
                    "accuracy": summary["accuracy"],
                    "accuracy_ci_low": ci_low,
                    "accuracy_ci_high": ci_high,
                    "macro_f1": summary["macro_f1"],
                }
            )
    plot_robustness(split_rows, args.output_dir / "robustness_by_corruption.png")

    metrics = {
        "clean": clean_summary,
        "calibrated_clean": calibrated_summary,
        "temperature": float(scaler.temperature.detach().cpu().item()),
        "splits": split_rows,
    }
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame(split_rows).to_csv(args.output_dir / "robustness_by_corruption.csv", index=False)


def main() -> None:
    evaluate(parse_args())


if __name__ == "__main__":
    main()
