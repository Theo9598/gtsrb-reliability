from __future__ import annotations

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def accuracy_from_logits(logits: np.ndarray, labels: np.ndarray) -> float:
    preds = np.argmax(logits, axis=1)
    return float(np.mean(preds == labels))


def topk_predictions(logits: np.ndarray, k: int = 5) -> tuple[np.ndarray, np.ndarray]:
    probs = softmax(logits)
    top_idx = np.argsort(-probs, axis=1)[:, :k]
    top_probs = np.take_along_axis(probs, top_idx, axis=1)
    return top_idx, top_probs


def expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == labels
    ece = 0.0
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    for lower, upper in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (confidences > lower) & (confidences <= upper)
        prop = np.mean(in_bin)
        if prop > 0:
            ece += abs(np.mean(accuracies[in_bin]) - np.mean(confidences[in_bin])) * prop
    return float(ece)


def reliability_bins(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> list[dict[str, float]]:
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = predictions == labels
    rows: list[dict[str, float]] = []
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    for lower, upper in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (confidences > lower) & (confidences <= upper)
        count = int(np.sum(in_bin))
        rows.append(
            {
                "bin_lower": float(lower),
                "bin_upper": float(upper),
                "count": count,
                "accuracy": float(np.mean(accuracies[in_bin])) if count else 0.0,
                "confidence": float(np.mean(confidences[in_bin])) if count else 0.0,
            }
        )
    return rows

