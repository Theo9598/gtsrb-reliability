# Appendix: GTSRB Reliability Workbench

This appendix contains the reproducibility details, commands, and diagnostic assets that are kept in the GitHub repository instead of the short final report.

## Repository Structure

- `src/gtsrb_robustness/`: training, evaluation, models, metrics, calibration, corruptions, and Grad-CAM utilities.
- `app/`: Streamlit demo code.
- `tests/`: smoke tests for datasets, model forward passes, metrics, and app helpers.
- `runs/`: local training logs and checkpoints.
- `outputs/`: evaluation metrics and report-ready figures.
- `reports/`: final report document.

## Models

The course-aligned model is `BaselineCNN` in `src/gtsrb_robustness/models.py`. It is trained from scratch and does not use pretrained weights.

Architecture:

| Stage | Layer(s) | Output | Course concept |
|---|---|---:|---|
| Input | RGB image resized to 96 x 96 | 3 channels | Image tensors |
| Block 1 | Conv2d 3->32, BatchNorm, ReLU, MaxPool | 32 maps | Filters, activation, pooling |
| Block 2 | Conv2d 32->64, BatchNorm, ReLU, MaxPool | 64 maps | Hierarchical features |
| Block 3 | Conv2d 64->128, BatchNorm, ReLU, MaxPool | 128 maps | Deeper visual patterns |
| Block 4 | Conv2d 128->256, BatchNorm, ReLU, AdaptiveAvgPool | 256 features | Global aggregation |
| Classifier | Flatten, Dropout p=0.35, Linear 256->43 | 43 logits | Regularization and multiclass classification |

The advanced comparison model is ResNet18 with ImageNet pretrained weights, RandAugment, and MixUp.

## Training Commands

Custom CNN:

```powershell
python -m gtsrb_robustness.train --model baseline_cnn --epochs 12 --batch-size 128 --output-dir runs/baseline_cnn
```

CNN ablations:

```powershell
python -m gtsrb_robustness.train --model baseline_cnn_no_bn --epochs 12 --batch-size 128 --output-dir runs/baseline_cnn_no_bn
python -m gtsrb_robustness.train --model baseline_cnn_no_dropout --epochs 12 --batch-size 128 --output-dir runs/baseline_cnn_no_dropout
```

Main robust ResNet18:

```powershell
python -m gtsrb_robustness.train --model resnet18 --epochs 25 --batch-size 96 --use-randaugment --mixup-alpha 0.2 --output-dir runs/resnet18_robust
```

## Evaluation Commands

ResNet18 final evaluation:

```powershell
python -m gtsrb_robustness.evaluate --checkpoint runs/resnet18_robust/best.pt --model resnet18 --output-dir outputs/resnet18_robust --num-workers 0
```

Custom CNN and ablation evaluation:

```powershell
python -m gtsrb_robustness.evaluate --checkpoint runs/baseline_cnn/best.pt --model baseline_cnn --output-dir outputs/baseline_cnn --num-workers 0 --skip-corruptions
python -m gtsrb_robustness.evaluate --checkpoint runs/baseline_cnn_no_bn/best.pt --model baseline_cnn_no_bn --output-dir outputs/baseline_cnn_no_bn --num-workers 0 --skip-corruptions
python -m gtsrb_robustness.evaluate --checkpoint runs/baseline_cnn_no_dropout/best.pt --model baseline_cnn_no_dropout --output-dir outputs/baseline_cnn_no_dropout --num-workers 0 --skip-corruptions
```

## Main Results

| Model | Best validation accuracy | Clean test accuracy | Clean test macro F1 |
|---|---:|---:|---:|
| Custom CNN | 78.03% | 58.99% | 39.26% |
| CNN without BatchNorm | 20.85% | 19.21% | 6.02% |
| CNN without Dropout | 50.03% | 39.05% | 21.06% |
| ResNet18 + RandAugment/MixUp | 99.97% | 98.76% | 98.33% |

Calibration for the final ResNet18 reduced expected calibration error from 6.36% to 3.04%.

## Diagnostic Figures

Core figures:

- `outputs/model_comparison_ablation.png`
- `outputs/resnet18_robust/training_curves.png`
- `outputs/resnet18_robust/robustness_by_corruption.png`
- `outputs/resnet18_robust/reliability_diagram.png`

Additional appendix diagnostics:

- `outputs/resnet18_robust/confusion_matrix.png`
- `outputs/resnet18_robust/error_examples.png`

## Demo

Local demo:

```powershell
streamlit run app/streamlit_app.py
```

Hosted demo:

https://huggingface.co/spaces/Theo9598/gtsrb-reliability-workbench

The demo is for educational use only and is not validated for real autonomous-driving, traffic-enforcement, or safety-critical deployment.
