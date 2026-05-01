from pathlib import Path
import sys

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gtsrb_robustness.models import build_model


def test_baseline_forward_pass():
    model = build_model("baseline_cnn", num_classes=43, pretrained=False)
    x = torch.randn(2, 3, 96, 96)
    logits = model(x)
    assert logits.shape == (2, 43)

