---
title: GTSRB Reliability Workbench
sdk: streamlit
sdk_version: 1.56.0
app_file: app.py
pinned: false
---

# GTSRB Robust Traffic Sign Recognition

This repo implements a course-project-ready pipeline for GTSRB traffic sign classification with robustness evaluation, confidence calibration, Grad-CAM explanations, and a Streamlit demo.

The project story is: a traffic sign classifier should be accurate on clean images, but also transparent about how it behaves under blur, noise, JPEG compression, and other realistic corruptions.

## What is included

- Baseline CNN trained from scratch.
- Transfer-learning model using ResNet18 or EfficientNet-B0.
- RandAugment and MixUp for robust training.
- Clean and corrupted test evaluation.
- Temperature scaling and expected calibration error.
- Grad-CAM visualization for model explanations.
- Streamlit app for upload, prediction, corruption simulation, and Grad-CAM.
- Report and presentation outlines aligned with the class project description.

The reproducibility appendix, including model architecture, training commands, ablation commands, and additional diagnostic figures, is in [`APPENDIX.md`](APPENDIX.md).

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you are installing PyTorch for a local RTX 2080 Ti, follow the current command from the official PyTorch selector for your CUDA version, then install the remaining packages from `requirements.txt`.

## Data

The training scripts use the Hugging Face dataset `tanganke/gtsrb`. The clean splits are `train` and `test`; corrupted evaluation splits are configured in `src/gtsrb_robustness/config.py`.

```powershell
python -m gtsrb_robustness.explore_data --output-dir outputs/eda
```

## Training

Fast baseline:

```powershell
python -m gtsrb_robustness.train --model baseline_cnn --epochs 12 --batch-size 128 --output-dir runs/baseline_cnn
```

Main model:

```powershell
python -m gtsrb_robustness.train --model resnet18 --epochs 25 --batch-size 96 --use-randaugment --mixup-alpha 0.2 --output-dir runs/resnet18_robust
```

Optional EfficientNet-B0:

```powershell
python -m gtsrb_robustness.train --model efficientnet_b0 --epochs 25 --batch-size 64 --use-randaugment --mixup-alpha 0.2 --output-dir runs/efficientnet_b0_robust
```

## Evaluation

```powershell
python -m gtsrb_robustness.evaluate --checkpoint runs/resnet18_robust/best.pt --model resnet18 --output-dir outputs/resnet18_robust
```

This writes metrics, confusion matrix, robustness chart data, reliability diagram data, and selected error examples.

Create report-ready curves and a markdown summary:

```powershell
python -m gtsrb_robustness.plot_history --history runs/resnet18_robust/history.csv --output outputs/resnet18_robust/training_curves.png
python -m gtsrb_robustness.summarize_results --metrics outputs/resnet18_robust/metrics.json --history runs/resnet18_robust/history.csv --output reports/results_summary.md
```

## Demo

```powershell
streamlit run app/streamlit_app.py
```

For Hugging Face Spaces, set the Space SDK to Streamlit and use `app/streamlit_app.py` as the entry file. Include a trained checkpoint in the Space or configure `GTSRB_CHECKPOINT`.

## Report assets checklist

- `outputs/eda/class_distribution.png`
- `outputs/eda/sample_grid.png`
- `runs/*/history.csv`
- `outputs/*/confusion_matrix.png`
- `outputs/*/robustness_by_corruption.png`
- `outputs/*/reliability_diagram.png`
- `outputs/*/error_examples.png`
