from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a compact report-ready results summary.")
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/results_summary.md"))
    return parser.parse_args()


def fmt_pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def summarize(metrics_path: Path, history_path: Path, output_path: Path) -> None:
    metrics = json.loads(metrics_path.read_text())
    history = pd.read_csv(history_path)
    best_epoch = history.sort_values("val_accuracy", ascending=False).iloc[0]
    final_epoch = history.iloc[-1]
    splits = pd.DataFrame(metrics["splits"])

    lines = [
        "# Results Summary",
        "",
        "## Training Diagnostics",
        "",
        f"- Best validation epoch: {int(best_epoch['epoch'])}",
        f"- Best validation accuracy: {fmt_pct(float(best_epoch['val_accuracy']))}",
        f"- Final train loss / validation loss: {final_epoch['train_loss']:.4f} / {final_epoch['val_loss']:.4f}",
        "",
        "## Clean Test",
        "",
        f"- Accuracy: {fmt_pct(metrics['clean']['accuracy'])}",
        f"- Macro F1: {fmt_pct(metrics['clean']['macro_f1'])}",
        f"- Expected calibration error before calibration: {fmt_pct(metrics['clean']['ece'])}",
        f"- Expected calibration error after temperature scaling: {fmt_pct(metrics['calibrated_clean']['ece'])}",
        f"- Learned temperature: {metrics['temperature']:.4f}",
        "",
        "## Robustness",
        "",
        "| Split | Source | Accuracy | 95% CI | Macro F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for _, row in splits.iterrows():
        ci = f"{fmt_pct(row['accuracy_ci_low'])}-{fmt_pct(row['accuracy_ci_high'])}" if "accuracy_ci_low" in row else ""
        lines.append(
            f"| {row['split']} | {row.get('source', '')} | {fmt_pct(row['accuracy'])} | {ci} | {fmt_pct(row['macro_f1'])} |"
        )

    lines.extend(
        [
            "",
            "## Report Interpretation Prompts",
            "",
            "- Compare the baseline CNN to the fine-tuned model to show the value of transfer learning.",
            "- Discuss the largest corruption-induced accuracy drop as the main reliability limitation.",
            "- Use the reliability diagram to explain whether confidence is trustworthy after calibration.",
            "- Tie the Grad-CAM examples to whether the model attends to the traffic sign rather than background artifacts.",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    summarize(args.metrics, args.history, args.output)


if __name__ == "__main__":
    main()

