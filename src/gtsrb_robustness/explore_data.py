from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .data import label_from_example, load_hf_gtsrb, pil_from_example
from .labels import GTSRB_LABELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create EDA figures for the GTSRB report.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/eda"))
    parser.add_argument("--max-samples", type=int, default=2000)
    return parser.parse_args()


def make_class_distribution(train_ds, output_dir: Path) -> None:
    labels = [label_from_example(train_ds[i]) for i in range(len(train_ds))]
    counts = pd.Series(labels).value_counts().sort_index()
    frame = pd.DataFrame({"class_id": counts.index, "count": counts.values})
    frame["class_name"] = frame["class_id"].map(GTSRB_LABELS)
    frame.to_csv(output_dir / "class_distribution.csv", index=False)
    plt.figure(figsize=(12, 5))
    sns.barplot(data=frame, x="class_id", y="count", color="#2f6fed")
    plt.title("GTSRB training class distribution")
    plt.xlabel("Class ID")
    plt.ylabel("Image count")
    plt.tight_layout()
    plt.savefig(output_dir / "class_distribution.png", dpi=220)
    plt.close()


def make_size_distribution(train_ds, output_dir: Path, max_samples: int) -> None:
    rows = []
    for i in range(min(len(train_ds), max_samples)):
        image = pil_from_example(train_ds[i])
        rows.append({"width": image.width, "height": image.height, "area": image.width * image.height})
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "image_size_distribution.csv", index=False)
    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=frame, x="width", y="height", s=14, alpha=0.35)
    plt.title("GTSRB image sizes")
    plt.tight_layout()
    plt.savefig(output_dir / "image_size_distribution.png", dpi=220)
    plt.close()


def make_sample_grid(train_ds, output_dir: Path) -> None:
    seen = set()
    examples = []
    for i in range(len(train_ds)):
        label = label_from_example(train_ds[i])
        if label not in seen:
            seen.add(label)
            examples.append((pil_from_example(train_ds[i]), label))
        if len(examples) == 43:
            break
    cols = 7
    rows = 7
    plt.figure(figsize=(12, 12))
    for idx, (image, label) in enumerate(examples):
        ax = plt.subplot(rows, cols, idx + 1)
        ax.imshow(image)
        ax.set_title(str(label), fontsize=8)
        ax.axis("off")
    plt.suptitle("One sample per GTSRB class", y=0.995)
    plt.tight_layout()
    plt.savefig(output_dir / "sample_grid.png", dpi=220)
    plt.close()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_ds = load_hf_gtsrb("train")
    make_class_distribution(train_ds, args.output_dir)
    make_size_distribution(train_ds, args.output_dir, args.max_samples)
    make_sample_grid(train_ds, args.output_dir)


if __name__ == "__main__":
    main()

