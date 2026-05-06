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
- Custom CNN robust-training comparison with RandAugment and MixUp.
- Transfer-learning model using ResNet18.
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

The training scripts use the Hugging Face dataset [`tanganke/gtsrb`](https://huggingface.co/datasets/tanganke/gtsrb). The clean splits are `train` and `test`; corrupted evaluation splits are configured in `src/gtsrb_robustness/config.py`.

For the reported comparisons, all models use the same stratified split of the original training data: 22,644 training images and 3,996 validation images with `seed=242` and `val_fraction=0.15`. The official 12,630-image `test` split is used only for final evaluation.

```powershell
python -m gtsrb_robustness.explore_data --output-dir outputs/eda
```

## Training

Fast baseline, main 25-epoch comparison run:

```powershell
python -m gtsrb_robustness.train --model baseline_cnn --epochs 25 --batch-size 128 --output-dir runs/baseline_cnn_25 --num-workers 0
```

Custom CNN with robust training, main 25-epoch comparison run:

```powershell
python -m gtsrb_robustness.train --model baseline_cnn --epochs 25 --batch-size 128 --use-randaugment --mixup-alpha 0.2 --output-dir runs/baseline_cnn_augmix_25 --num-workers 0
```

Main model:

```powershell
python -m gtsrb_robustness.train --model resnet18 --epochs 25 --batch-size 96 --use-randaugment --mixup-alpha 0.2 --output-dir runs/resnet18_robust
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

## Final Report Assets

- `reports/submission_report_final_cnn_ablation.pdf`
- `outputs/cnn_augmix_resnet_comparison.png`
- `outputs/resnet18_robust/training_curves.png`
- `outputs/resnet18_robust/robustness_by_corruption.png`
- `outputs/resnet18_robust/reliability_diagram.png`
