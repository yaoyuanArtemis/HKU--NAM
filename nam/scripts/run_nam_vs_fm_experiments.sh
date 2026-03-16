#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$ROOT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Current experiments/nam_vs_fm/run.py maps to the paper-style settings as follows:
# - output penalty λ1 -> --output_regularization
# - weight decay λ2 -> --l2_regularization
# - hidden layers 64-64-32 -> --shallow=False with --num_basis_functions=64
# - hidden units 1024 + ReLU-1 + ExU -> --activation=exu with --num_basis_functions=1024
# - Regression datasets need --regression=True
#
# Note:
# The current implementation does not support setting every feature net to an
# exact fixed width directly. Widths are inferred by:
# min(num_basis_functions, unique_values * units_multiplier)
# so the commands below are the closest match under the current codebase.

PYTHONPATH=. python experiments/nam_vs_fm/run.py \
  --dataset_name=Fico \
  --regression=True \
  --learning_rate=0.0161 \
  --output_regularization=0.0205 \
  --l2_regularization=1.07e-5 \
  --dropout=0.0 \
  --feature_dropout=0.0 \
  --num_basis_functions=64 \
  --units_multiplier=2 \
  --activation=relu \
  --shallow=False \
  --output_dir="$REPO_ROOT/outputs/nam_vs_fm/Fico"

PYTHONPATH=. python experiments/nam_vs_fm/run.py \
  --dataset_name=Housing \
  --regression=True \
  --learning_rate=0.00674 \
  --output_regularization=0.001 \
  --l2_regularization=1e-6 \
  --dropout=0.0 \
  --feature_dropout=0.0 \
  --num_basis_functions=64 \
  --units_multiplier=2 \
  --activation=relu \
  --shallow=False \
  --output_dir="$REPO_ROOT/outputs/nam_vs_fm/Housing"

PYTHONPATH=. python experiments/nam_vs_fm/run.py \
  --dataset_name=Credit \
  --regression=False \
  --learning_rate=0.0157 \
  --output_regularization=0.0 \
  --l2_regularization=4.95e-6 \
  --dropout=0.8 \
  --feature_dropout=0.0 \
  --num_basis_functions=64 \
  --n_models=5 \
  --units_multiplier=2 \
  --activation=exu \
  --shallow=True \
  --output_dir="$REPO_ROOT/outputs/nam_vs_fm/Credit"
