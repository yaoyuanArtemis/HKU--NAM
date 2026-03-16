#!/usr/bin/env python
# coding=utf-8
"""Training script for Neural Additive Models implemented with PyTorch."""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys
from typing import Iterable, Tuple

os.environ.setdefault("KMP_USE_SHM", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")


def _repo_root() -> str:
  return osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))


def _add_repo_parent_to_path(repo_root: str) -> None:
  parent = osp.dirname(repo_root)
  if parent not in sys.path:
    sys.path.insert(0, parent)


_add_repo_parent_to_path(_repo_root())

from neural_additive_models.runtime import seed_everything
from neural_additive_models.training.trainer import TrainingConfig
from neural_additive_models.training.trainer import train_ensemble
from neural_additive_models import data_utils

DatasetType = data_utils.DatasetType
N_FOLDS = 5


def str2bool(value):
  """Parse bool-like CLI values."""
  if isinstance(value, bool):
    return value
  normalized = str(value).strip().lower()
  if normalized in {"true", "1", "yes", "y"}:
    return True
  if normalized in {"false", "0", "no", "n"}:
    return False
  raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def build_parser() -> argparse.ArgumentParser:
  """Create the training argument parser."""
  parser = argparse.ArgumentParser(description="Train NAM or DNN models with PyTorch.")
  parser.add_argument("--training_epochs", type=int, default = 5)
  parser.add_argument("--learning_rate", type=float, default=1e-2)
  parser.add_argument("--output_regularization", type=float, default=0.0)
  parser.add_argument("--l2_regularization", type=float, default=0.0)
  parser.add_argument("--batch_size", type=int, default=1024)
  parser.add_argument("--logdir", type=str, default=osp.join(_repo_root(), "outputs", "nam", "training"))
  parser.add_argument("--dataset_name", type=str, default="Teleco")
  parser.add_argument("--decay_rate", type=float, default=0.995)
  parser.add_argument("--dropout", type=float, default=0.5)
  parser.add_argument("--data_split", type=int, default=1)
  parser.add_argument("--tf_seed", type=int, default=1)
  parser.add_argument("--feature_dropout", type=float, default=0.0)
  parser.add_argument("--num_basis_functions", type=int, default=32)
  parser.add_argument("--units_multiplier", type=int, default=1)
  parser.add_argument("--cross_val", type=str2bool, nargs="?", const=True, default=False)
  parser.add_argument("--max_checkpoints_to_keep", type=int, default=1)
  parser.add_argument("--save_checkpoint_every_n_epochs", type=int, default=10)
  parser.add_argument("--n_models", type=int, default=1)
  parser.add_argument("--num_splits", type=int, default=3)
  parser.add_argument("--fold_num", type=int, default=1)
  parser.add_argument("--activation", type=str, default="exu")
  parser.add_argument("--regression", type=str2bool, nargs="?", const=True, default=False)
  parser.add_argument("--debug", type=str2bool, nargs="?", const=True, default=False)
  parser.add_argument("--shallow", type=str2bool, nargs="?", const=True, default=False)
  parser.add_argument("--use_dnn", type=str2bool, nargs="?", const=True, default=False)
  parser.add_argument("--early_stopping_epochs", type=int, default=60)
  parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"])
  return parser


def config_from_args(args: argparse.Namespace) -> TrainingConfig:
  """Convert parsed CLI values into a training config."""
  return TrainingConfig(
      training_epochs=args.training_epochs,
      learning_rate=args.learning_rate,
      output_regularization=args.output_regularization,
      l2_regularization=args.l2_regularization,
      batch_size=args.batch_size,
      decay_rate=args.decay_rate,
      dropout=args.dropout,
      feature_dropout=args.feature_dropout,
      num_basis_functions=args.num_basis_functions,
      units_multiplier=args.units_multiplier,
      n_models=args.n_models,
      activation=args.activation,
      regression=args.regression,
      debug=args.debug,
      shallow=args.shallow,
      use_dnn=args.use_dnn,
      early_stopping_epochs=args.early_stopping_epochs,
      save_checkpoint_every_n_epochs=args.save_checkpoint_every_n_epochs,
      max_checkpoints_to_keep=args.max_checkpoints_to_keep,
      device=args.device,
      tf_seed=args.tf_seed,
  )


def training(
    x_train,
    y_train,
    x_validation,
    y_validation,
    logdir,
    config: TrainingConfig,
    x_test=None,
    y_test=None,
    return_test_metric: bool = False,
):
  """Train the configured ensemble and return aggregate metrics."""
  seed_everything(config.tf_seed)
  print(f"Started training with logdir {logdir}")
  result = train_ensemble(
      x_train=x_train,
      y_train=y_train,
      x_validation=x_validation,
      y_validation=y_validation,
      logdir=logdir,
      config=config,
      x_test=x_test,
      y_test=y_test,
      return_test_metric=return_test_metric,
  )
  print("Finished training.")
  return result


def create_test_train_fold(
    dataset_name: str,
    regression: bool,
    fold_num: int,
    num_folds: int = N_FOLDS,
    num_splits: int = 3,
):
  """Split the dataset into training and held-out test sets."""
  data_x, data_y, _ = data_utils.load_dataset(dataset_name)
  print(f"Dataset: {dataset_name}, Size: {data_x.shape[0]}")
  print(f"Cross-val fold: {fold_num}/{num_folds}")
  (x_train_all, y_train_all), test_dataset = data_utils.get_train_test_fold(
      data_x,
      data_y,
      fold_num=fold_num,
      num_folds=num_folds,
      stratified=not regression,
  )
  data_gen = data_utils.split_training_dataset(
      x_train_all,
      y_train_all,
      n_splits=num_splits,
      stratified=not regression,
  )
  return data_gen, test_dataset


def write_training_params(args: argparse.Namespace, fold_dir: str) -> None:
  """Persist CLI parameters for later evaluation and plotting."""
  os.makedirs(fold_dir, exist_ok=True)
  params_path = osp.join(fold_dir, "training_params.json")
  with open(params_path, "w", encoding="utf-8") as f:
    json.dump(vars(args), f, indent=2)


def single_split_training(
    data_gen: Iterable[Tuple[DatasetType, DatasetType]],
    logdir: str,
    args: argparse.Namespace,
):
  """Use a specific train/validation split for model training."""
  split = None
  for _ in range(args.data_split):
    split = next(data_gen)
  if split is None:
    raise ValueError("No training split was produced.")
  (x_train, y_train), (x_validation, y_validation) = split
  fold_dir = osp.join(logdir, f"fold_{args.fold_num}")
  write_training_params(args, fold_dir)
  curr_logdir = osp.join(logdir, f"fold_{args.fold_num}", f"split_{args.data_split}")
  return training(
      x_train=x_train,
      y_train=y_train,
      x_validation=x_validation,
      y_validation=y_validation,
      logdir=curr_logdir,
      config=config_from_args(args),
  )


def main(argv=None) -> None:
  """CLI entrypoint."""
  parser = build_parser()
  args = parser.parse_args(argv)
  if args.data_split != 1 and args.cross_val:
    raise ValueError("Data split should not be used together with cross validation.")
  data_x, data_y, _ = data_utils.load_dataset(args.dataset_name)
  print(f"Dataset: {args.dataset_name}, Size: {data_x.shape[0]}")
  print(f"Cross-val fold: {args.fold_num}/{N_FOLDS}")
  (x_train_all, y_train_all), _ = data_utils.get_train_test_fold(
      data_x,
      data_y,
      fold_num=args.fold_num,
      num_folds=N_FOLDS,
      stratified=not args.regression,
  )
  data_gen = data_utils.split_training_dataset(
      x_train_all,
      y_train_all,
      n_splits=args.num_splits,
      stratified=not args.regression,
  )
  single_split_training(data_gen, args.logdir, args)


if __name__ == "__main__":
  main()
