from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gtsrb_robustness.corruptions import DEMO_CORRUPTIONS, EVAL_CORRUPTIONS
from gtsrb_robustness.labels import GTSRB_LABELS, NUM_CLASSES, label_name
from gtsrb_robustness.metrics import accuracy_from_logits, expected_calibration_error, softmax, topk_predictions


def test_label_map_has_43_classes():
    assert NUM_CLASSES == 43
    assert len(GTSRB_LABELS) == 43
    assert label_name(14) == "Stop"


def test_metrics_are_well_formed():
    logits = np.array([[4.0, 1.0, 0.2], [0.1, 0.5, 2.2], [0.0, 3.0, 1.0]])
    labels = np.array([0, 2, 1])
    probs = softmax(logits)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert accuracy_from_logits(logits, labels) == 1.0
    assert 0.0 <= expected_calibration_error(probs, labels, n_bins=5) <= 1.0
    top_idx, top_probs = topk_predictions(logits, k=2)
    assert top_idx.shape == (3, 2)
    assert top_probs.shape == (3, 2)


def test_demo_corruptions_preserve_image_contract():
    image = Image.new("RGB", (64, 48), color=(120, 40, 20))
    for fn in {**DEMO_CORRUPTIONS, **EVAL_CORRUPTIONS}.values():
        corrupted = fn(image)
        assert corrupted.mode == "RGB"
        assert corrupted.size == image.size
