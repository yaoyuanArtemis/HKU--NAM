# Neural Additive Models in PyTorch

This repository now uses the repository root as the main project entrypoint. It contains:

- The full PyTorch NAM implementation under `nam/`
- Root-level NAM entry scripts such as `nam_train.py`
- Baseline comparison utilities under `baseline/`
- Batch workflows for reproducing NAM-style experiments across datasets

## Repository Layout

```text
baseline/                     # baseline model comparison utilities
nam/                          # NAM source package
neural_additive_models/       # compatibility package for root-level execution
nam_train.py                  # root NAM training entrypoint
nam_evaluate.py               # root NAM evaluation entrypoint
nam_plot_ensemble.py          # root NAM visualization entrypoint
nam_compas_multitask.py       # COMPAS single-task vs multitask entrypoint
nam_vs_fm.py                  # NAM vs NAM+FM entrypoint
main.py                       # batch baseline / NAM workflow
download_datasets.py          # dataset preparation helper
NAM_Complete_Workflow.ipynb   # notebook workflow
```

## Installation

Create an environment and install the project dependencies:

```bash
pip install -r requirements.txt
```

Optional editable install with CLI entrypoints:

```bash
pip install -e .
```

After editable install, the following commands are available:

- `nam-batch`
- `nam-compare`
- `nam-train`
- `nam-evaluate`
- `nam-plot`
- `nam-compas`
- `nam-vs-fm`

## Datasets

The shared preprocessing is implemented in `nam/data_utils.py`. The workflow supports `Housing`, `Fico`, `Credit`, `Adult`, `Telco`, `BreastCancer`, `Heart`, `Mimic2`, and `Recidivism`.

Prepare local datasets with:

```bash
python download_datasets.py
```

The script can automatically prepare public datasets and will print instructions for datasets that require manual download or access approval.

## NAM Experiments

### 1. Base NAM Reproduction

Train a NAM ensemble from the repository root:

```bash
python nam_train.py \
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
python nam_evaluate.py \
  --run_dir outputs/nam/housing_example/training/fold_1
```

Plot ensemble shape functions:

```bash
python nam_plot_ensemble.py \
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

```bash
python nam_compas_multitask.py --mode cv --n_models 20 --training_epochs 50
python nam_compas_multitask.py --mode figure --n_models 20 --training_epochs 50
python nam_compas_multitask.py --mode all --n_models 100 --training_epochs 80
```

Outputs are written under `outputs/compas_multitask/`.

### 3. NAM vs NAM+FM

```bash
python nam_vs_fm.py \
  --dataset_name Credit \
  --regression False \
  --n_models 5 \
  --training_epochs 1000 \
  --fm_rank 12
```

Or run the prepared script:

```bash
bash nam/scripts/run_nam_vs_fm_experiments.sh
```

Outputs are written under `outputs/nam_vs_fm/`.

## Baseline Comparison

### Single Dataset Comparison

Use the baseline comparison entrypoint:

```bash
python baseline/run_experiment.py \
  --data_path datasets/breast_cancer.csv \
  --target_column target \
  --task classification \
  --output_dir comparison_results/breast_cancer
```

This runs the available baseline models and records a markdown report plus CSV summary.

### Batch Baseline + NAM Workflow

The repository root `main.py` runs the multi-dataset workflow:

```bash
python main.py
python main.py --train_nam
python main.py --only_nam
```

- `python main.py`: baseline models only
- `python main.py --train_nam`: baselines followed by NAM training
- `python main.py --only_nam`: NAM training only

Batch outputs are written under `all_results/`, and NAM logs are stored under `all_results/nam_logs/`.

## Notes on Paths

- The canonical user-facing entrypoints now live at the repository root.
- The actual NAM implementation remains under `nam/`.
- The `neural_additive_models` package exists as a compatibility layer so both editable installs and root-level scripts work consistently.
- The recommended shared output root for standalone NAM experiments is `outputs/`.

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

## Benchmark Results

The following tables summarize representative benchmark results for NAM, NAMFM, and a set of standard baseline models. Reported values follow the evaluation metrics used in each dataset, and ensemble-based NAM variants are shown as mean ± standard deviation when multiple runs were aggregated.

### California

| Model | Num Parameters | Train RMSE | Val RMSE | Test RMSE |
| --- | ---: | ---: | ---: | ---: |
| NAM | 51,201 | 0.5520 | 0.5755 | 0.581 ± 0.009 |
| EBM | 800 | 0.375735 | 0.490463 | 0.465602 |
| XGBoost | 3,100 | 0.410846 | 0.503544 | 0.492333 |
| DNN-MLP | 11,521 | 0.484241 | 0.541413 | 0.526535 |
| CART | 61 | 0.68923 | 0.73501 | 0.732572 |
| Linear | 9 | 0.716174 | 0.733898 | 0.744629 |
| NAMFM | 51,297 | 0.5729 ± 0.0118 | 0.5683 ± 0.0154 | 0.5782 ± 0.0129 |

### Credit

| Model | Num Parameters | Train AUROC | Val AUROC | Test AUROC |
| --- | ---: | ---: | ---: | ---: |
| NAM | 92,161 | 0.9902 ± 0.0025 | 0.9765 ± 0.0152 | 0.973 ± 0.010 |
| EBM | 3,000 | 0.999988 | 0.964192 | 0.976328 |
| XGBoost | 3,100 | 0.999973 | 0.979355 | 0.974806 |
| DNN-MLP | 14,337 | 0.997847 | 0.932886 | 0.971306 |
| Logistic | 31 | 0.983137 | 0.964321 | 0.95368 |
| CART | 33 | 0.940962 | 0.877322 | 0.906417 |
| NAMFM | 92,521 | 0.9950 ± 0.0027 | 0.9802 ± 0.0136 | 0.9738 ± 0.0086 |

### Heloc

| Model | Num Parameters | Train RMSE | Val RMSE | Test RMSE |
| --- | ---: | ---: | ---: | ---: |
| NAM | 115,721 | 3.4063 ± 0.0531 | 3.6960 ± 0.1508 | 3.590 ± 0.179 |
| EBM | 2,300 | 2.66589 | 3.13885 | 3.13495 |
| XGBoost | 3,100 | 2.47542 | 3.17039 | 3.18007 |
| DNN-MLP | 13,441 | 2.89539 | 3.62039 | 3.73619 |
| Linear | 24 | 4.27581 | 4.38194 | 4.24178 |
| CART | 63 | 4.80589 | 4.82292 | 4.91923 |
| NAMFM | 115,981 | 3.3335 ± 0.0674 | 3.6595 ± 0.1717 | 3.5195 ± 0.1676 |

### COMPAS Single-Task, Multitask, and FMNAMs Comparison

The following comparison summarizes AUROC performance on the COMPAS benchmark under single-task and multitask NAM training, together with the combined-result FMNAMs variant.

| Model | COMPAS Women | COMPAS Men | COMPAS Combined |
| --- | ---: | ---: | ---: |
| Single-task NAM | 0.710 ± 0.044 | 0.730 ± 0.015 | 0.730 ± 0.012 |
| Multitask NAM | 0.711 ± 0.041 | 0.730 ± 0.016 | 0.730 ± 0.016 |
| FMNAMs | - | - | 0.729 ± 0.020 |
