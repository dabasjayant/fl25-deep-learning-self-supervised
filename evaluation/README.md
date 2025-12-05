# Evaluation Scripts

This directory contains reproducible evaluation utilities for a custom self-supervised ViT-based representation learner inspired by DINO, but implemented in-house. All checkpoints referenced here are produced by this custom training pipeline.

- Linear evaluation with a single linear head on frozen features, including systematic hyperparameter tuning.
- kNN-based evaluation with GridSearchCV to quickly assess representation quality and generate submissions.

Use these scripts after training to benchmark learned embeddings with rigorous, configurable tuning.

## Prerequisites
- Python 3.9+ (tested on macOS, zsh)
- PyTorch and torchvision installed
- Common ML packages: `numpy`, `scikit-learn`, `pandas`, `tqdm`
- Trained checkpoint(s) from the custom implementation (compatible with `VisionTransformer` backbone used here)
- Dataset prepared locally with images accessible on disk

If your environment isn’t set up yet, install minimal dependencies:

```zsh
python -m venv .venv
source .venv/bin/activate
pip install torch torchvision numpy scikit-learn pandas tqdm
```

## Data Layout Assumptions
- Dataset prepared per the released test set instructions.
- Reference: `https://github.com/tsb0601/fall2025_finalproject/tree/main`

Adjust the dataset root or label parsing if your layout differs.

## Scripts Overview

### 1) Linear Evaluation (`eval_linear.py` and `eval_linear_max.py`)
- Loads the custom-trained checkpoint and freezes the backbone.
- Extracts representations (typically the `CLS` token concatenated with average pooled patch tokens).
- Trains a linear classifier on top of frozen features, with systematic hyperparameter tuning.

Run basic linear eval:

```zsh
python evaluation/eval_linear.py --data_dir testset_1/dataset/data --checkpoint <CHECKPOINT_PATH>
```

Run the variant tuned for maximum validation performance (broader hyperparameter search):

```zsh
python evaluation/eval_linear_max.py --data_dir testset_1/dataset/data --checkpoint <CHECKPOINT_PATH>
```

Input parameters (argparse):
- `--data_dir`: dataset root (expects `train/`, `val/`, `test/` and CSVs).
- `--checkpoint`: path to the custom model checkpoint (`.pth`).
- `--output`: output CSV path. Defaults to:
  - `submission_linear.csv` in `eval_linear.py`,
  - `submission_linear_max.csv` in `eval_linear_max.py`.
- `--arch`: backbone size (`vit_base` or `vit_small`).
- `--batch_size`: feature precompute batch size.
- `--epochs_per_config`: only used for initial data loading.

Expected outputs:
- Best validation accuracy and the winning configuration.
- Leaderboard of top-n hyperparameter combinations.
- Saved submission CSV for the test split.

Hyperparameter tuning:
- `eval_linear.py`: grid over learning rate, weight decay, and epochs per configuration.
- `eval_linear_max.py`: broader grid including optimizer (`adamw`/`sgd`), batch size, learning rate (extended range), weight decay, and epochs.

### 2) kNN Evaluation and Submission (`create_submission_knn_tuned.py`)
- Computes embeddings with the custom-trained backbone.
- Applies k-Nearest Neighbors over train/val splits with `GridSearchCV`.
- Produces a submission file (CSV) and optional diagnostics (e.g., confusion matrix if enabled).

Run kNN submission generation:

```zsh
python evaluation/create_submission_knn_tuned.py --data_dir testset_1/dataset/data --checkpoint <CHECKPOINT_PATH>
```

Input parameters (argparse):
- `--data_dir`: dataset root (expects `train/`, `val/`, `test/` and CSVs).
- `--checkpoint`: path to the custom model checkpoint (`.pth`).
- `--output`: output CSV path (default `submission.csv`).
- `--arch`: backbone size (`vit_base` or `vit_small`).
- `--batch_size`: feature precompute batch size.
- `--num_workers`: data loader workers.

Expected outputs:
- A submission CSV with predicted labels for the evaluation split.
- Tuning leaderboard printed to stdout.
- Summary metrics on validation.

Hyperparameter tuning:
- `create_submission_knn_tuned.py`: grid over `n_neighbors`, `weights` (`uniform`/`distance`), and `p` (Manhattan vs Euclidean).

## Reproducibility Tips
- Fix random seeds in the scripts where available.
- Keep `torch.backends.cudnn.deterministic=True` if using CUDA.
- Record the exact `checkpoint_path` and git commit when evaluating.

## Troubleshooting
- Import errors: ensure your Python path includes the project root (run from repo root or add `PYTHONPATH=.`).
- Mismatched checkpoint: verify the model architecture matches your checkpoint.
- Dataset layout: confirm class names and folder structure align with the assumptions above.
- Performance variance: adjust `batch_size`, learning rate, or kNN `k` based on hardware and dataset size.

## Example End-to-End Flow
1. Train our model and obtain `checkpoint_{EPOCH}.pth`.
2. Configure paths and hyperparameters in `evaluation/eval_linear.py` if needed.
3. Run linear eval to get baseline accuracy.
4. Run `evaluation/create_submission_knn_tuned.py` to generate a kNN submission and quick metrics.
5. Iterate on hyperparameters (LR, epochs, k) to tune results.

## Notes
- These scripts are intentionally lightweight: edit in-place for your dataset and compute.
- For large-scale runs, consider exporting embeddings once and reusing them across evaluations.
