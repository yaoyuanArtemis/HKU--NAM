# Neural Additive Models in PyTorch

This repository is a PyTorch reproduction of the paper [Neural Additive Models: Interpretable Machine Learning with Neural Nets](https://arxiv.org/abs/2004.13912).

It contains a reusable NAM implementation plus three experiment groups:

- Base NAM training, evaluation, and ensemble plotting.
- COMPAS single-task vs multitask comparison.
- NAM vs NAM+FM comparison.

## Repository Layout

```text
experiments/
  nam/                 # base NAM train / evaluate / plot entrypoints
  compas_multitask/    # COMPAS single-task vs multitask experiment
  nam_vs_fm/           # NAM vs NAM+FM experiment
scripts/
  run_nam_vs_fm_experiments.sh
models/                # NAM model definitions
training/              # training loop, losses, metrics, data loaders
runtime/               # device, checkpoint, seed utilities
data_utils.py          # dataset loading and preprocessing
data/                  # local datasets
outputs/               # recommended output root for all experiments
```

## Installation

Create an environment and install the dependencies from `requirements.txt`.

```bash
pip install -r requirements.txt
```

## Datasets

The code supports the datasets wired in `data_utils.py`, including `Housing`, `Fico`, `Credit`, `Adult`, `Telco`, `BreastCancer`, `Heart`, `Mimic2`, and `Recidivism` (COMPAS).

Place local dataset files under `data/` using the names expected by `data_utils.py`. The original NAM paper datasets except MIMIC-II were released in the public GCP bucket `gs://nam_datasets/data`.

## Experiments

### 1. Base NAM Reproduction

Train a NAM ensemble:

```bash
PYTHONPATH=. python experiments/nam/train.py \
  --dataset_name Housing \
  --regression True \
  --n_models 5 \
  --num_splits 3 \
  --fold_num 1 \
  --training_epochs 1000 \
  --learning_rate 0.00674 \
  --activation relu \
  --shallow False \
  --num_basis_functions 64 \
  --output_regularization 0.001 \
  --l2_regularization 1e-6 \
  --dropout 0.0 \
  --feature_dropout 0.0 \
  --logdir outputs/nam/housing_example/training
```

Evaluate a trained fold:

```bash
PYTHONPATH=. python experiments/nam/evaluate.py \
  --run_dir outputs/nam/housing_example/training/fold_1
```

Plot ensemble shape functions:

```bash
PYTHONPATH=. python experiments/nam/plot_ensemble.py \
  --run_dir outputs/nam/housing_example/training/fold_1
```

Base NAM outputs follow this layout:

```text
outputs/nam/<run_name>/training/fold_<k>/split_<s>/model_<i>/
outputs/nam/<run_name>/training/fold_<k>/training_params.json
outputs/nam/<run_name>/training/fold_<k>/test_outputs/
outputs/nam/<run_name>/training/fold_<k>/visualization_outputs/
```

### 2. COMPAS Single-Task vs Multitask

Run 5-fold cross-validation:

```bash
PYTHONPATH=. python experiments/compas_multitask/run.py \
  --mode cv \
  --n_models 20 \
  --training_epochs 50
```

Generate the figure-oriented run:

```bash
PYTHONPATH=. python experiments/compas_multitask/run.py \
  --mode figure \
  --n_models 20 \
  --training_epochs 50
```

Run both:

```bash
PYTHONPATH=. python experiments/compas_multitask/run.py \
  --mode all \
  --n_models 100 \
  --training_epochs 80
```

Outputs are written under `outputs/compas_multitask/`.

### 3. NAM vs NAM+FM

Run a single comparison:

```bash
PYTHONPATH=. python experiments/nam_vs_fm/run.py \
  --dataset_name Credit \
  --regression False \
  --n_models 5 \
  --training_epochs 1000 \
  --fm_rank 12
```

Run the prepared shell script:

```bash
bash scripts/run_nam_vs_fm_experiments.sh
```

Outputs are written under `outputs/nam_vs_fm/`.

## Notes on Paths

- All experiment entrypoints now live under `experiments/`.
- Helper shell scripts live under `scripts/`.
- The recommended shared output root is `outputs/`.

## Citation

If you use this code in research, please cite:

```bibtex
@article{agarwal2021neural,
  title={Neural additive models: Interpretable machine learning with neural nets},
  author={Agarwal, Rishabh and Melnick, Levi and Frosst, Nicholas and Zhang, Xuezhou and Lengerich, Ben and Caruana, Rich and Hinton, Geoffrey E},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}
```

## COMPAS Disclaimer

The COMPAS dataset is included only as a reproduction example for interpretability and fairness-related experiments. Any criminal justice prediction task has significant ethical and social risks and should be treated accordingly.
