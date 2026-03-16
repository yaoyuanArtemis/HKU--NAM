#!/usr/bin/env python
# coding=utf-8
"""Standalone NAM ensemble testing script based on PyTorch checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys
from datetime import datetime
from typing import Any, Dict

import numpy as np

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

import torch

from neural_additive_models import data_utils
from neural_additive_models.experiments.nam.train import str2bool
from neural_additive_models.runtime import find_checkpoint_path
from neural_additive_models.runtime import load_checkpoint
from neural_additive_models.runtime import resolve_device
from neural_additive_models.training.metrics import calculate_metric
from neural_additive_models.training.trainer import TrainingConfig
from neural_additive_models.training.trainer import create_model


def batched_predict(model, x_data: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
  """Predict in batches."""
  model.eval()
  predictions = []
  with torch.no_grad():
    for start in range(0, x_data.shape[0], batch_size):
      batch = torch.as_tensor(x_data[start:start + batch_size], dtype=torch.float32, device=device)
      predictions.append(model(batch, training=False).detach().cpu().numpy())
  return np.concatenate(predictions, axis=0)


def _load_training_params(run_dir: str) -> Dict[str, Any]:
  """Load training params from the canonical or fallback path."""
  params_path = osp.join(run_dir, "training_params.json")
  if not osp.exists(params_path):
    fold_name = osp.basename(osp.normpath(run_dir))
    alt_params_path = osp.join(osp.dirname(run_dir), "training", fold_name, "training_params.json")
    if osp.exists(alt_params_path):
      params_path = alt_params_path
    else:
      raise FileNotFoundError(
          f"training_params.json not found in run_dir: {run_dir}. "
          "Pass explicit args or run training first."
      )
  with open(params_path, "r", encoding="utf-8") as f:
    return json.load(f)


def _infer_model_logdir(run_dir: str, fold_num: int, split_idx: int) -> str:
  """Resolve the directory that contains model subdirectories."""
  run_dir = osp.normpath(run_dir)
  parent = osp.dirname(run_dir)
  if osp.basename(parent) == "training":
    logdir_root = parent
  else:
    logdir_root = osp.join(parent, "training")
  return osp.join(logdir_root, f"fold_{fold_num}", f"split_{split_idx}")


def _config_from_values(values: Dict[str, Any]) -> TrainingConfig:
  """Create a training config from loaded values."""
  return TrainingConfig(
      training_epochs=int(values.get("training_epochs", 1)),
      learning_rate=float(values.get("learning_rate", 1e-2)),
      output_regularization=float(values.get("output_regularization", 0.0)),
      l2_regularization=float(values.get("l2_regularization", 0.0)),
      batch_size=int(values.get("batch_size", 1024)),
      decay_rate=float(values.get("decay_rate", 0.995)),
      dropout=float(values.get("dropout", 0.0)),
      feature_dropout=float(values.get("feature_dropout", 0.0)),
      num_basis_functions=int(values.get("num_basis_functions", 1000)),
      units_multiplier=int(values.get("units_multiplier", 2)),
      n_models=int(values.get("n_models", 1)),
      activation=str(values.get("activation", "exu")),
      regression=bool(values.get("regression", False)),
      debug=bool(values.get("debug", False)),
      shallow=bool(values.get("shallow", False)),
      use_dnn=bool(values.get("use_dnn", False)),
      early_stopping_epochs=int(values.get("early_stopping_epochs", 60)),
      save_checkpoint_every_n_epochs=int(values.get("save_checkpoint_every_n_epochs", 10)),
      max_checkpoints_to_keep=int(values.get("max_checkpoints_to_keep", 1)),
      device=str(values.get("device", "auto")),
      tf_seed=int(values.get("tf_seed", 1)),
  )


def build_parser() -> argparse.ArgumentParser:
  """Build the evaluation parser."""
  parser = argparse.ArgumentParser(description="Evaluate trained NAM ensemble on held-out test split.")
  parser.add_argument("--run_dir", default=None)
  parser.add_argument("--model_logdir", default=None)
  parser.add_argument("--dataset_name", default=None)
  parser.add_argument("--n_models", type=int, default=None)
  parser.add_argument("--fold_num", type=int, default=None)
  parser.add_argument("--num_folds", type=int, default=5)
  parser.add_argument("--num_splits", type=int, default=None)
  parser.add_argument("--split_idx", type=int, default=1)
  parser.add_argument("--activation", default=None)
  parser.add_argument("--num_basis_functions", type=int, default=None)
  parser.add_argument("--units_multiplier", type=int, default=None)
  parser.add_argument("--dropout", type=float, default=None)
  parser.add_argument("--feature_dropout", type=float, default=None)
  parser.add_argument("--shallow", type=str2bool, nargs="?", const=True, default=None)
  parser.add_argument("--regression", type=str2bool, nargs="?", const=True, default=None)
  parser.add_argument("--use_dnn", type=str2bool, nargs="?", const=True, default=None)
  parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
  parser.add_argument("--output_dir", default=None)
  return parser


def main(argv=None):
  """CLI entrypoint."""
  parser = build_parser()
  args = parser.parse_args(argv)
  cfg = _load_training_params(args.run_dir) if args.run_dir else {}
  merged = {**cfg}
  for key, value in vars(args).items():
    if value is not None:
      merged[key] = value
  dataset_name = merged.get("dataset_name")
  if not dataset_name:
    raise ValueError("Missing dataset_name. Pass --dataset_name or --run_dir.")
  n_models = int(merged.get("n_models", 1))
  fold_num = int(merged.get("fold_num", 1))
  num_splits = int(merged.get("num_splits", 3))
  split_idx = int(merged.get("split_idx", 1))
  config = _config_from_values(merged)
  config.device = args.device
  model_logdir = args.model_logdir or _infer_model_logdir(args.run_dir, fold_num, split_idx)
  output_dir = args.output_dir or (
      osp.join(args.run_dir, "test_outputs")
      if args.run_dir else osp.join(_repo_root(), "outputs", "nam", "evaluation"))
  os.makedirs(output_dir, exist_ok=True)
  device = resolve_device(config.device)

  data_x, data_y, _ = data_utils.load_dataset(dataset_name)
  (x_train_all, y_train_all), test_dataset = data_utils.get_train_test_fold(
      data_x,
      data_y,
      fold_num=fold_num,
      num_folds=args.num_folds,
      stratified=not config.regression,
  )
  data_gen = data_utils.split_training_dataset(
      x_train_all,
      y_train_all,
      n_splits=num_splits,
      stratified=not config.regression,
  )
  split = None
  for _ in range(split_idx):
    split = next(data_gen)
  (x_train, _), _ = split
  x_test, y_test = test_dataset

  per_model_metric = []
  for model_idx in range(n_models):
    checkpoint_path = find_checkpoint_path(osp.join(model_logdir, f"model_{model_idx}"))
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    model = create_model(config, x_train).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    y_pred = batched_predict(model, x_test, batch_size=config.batch_size, device=device)
    per_model_metric.append(float(calculate_metric(y_test, y_pred, regression=config.regression)))

  metric_name = "RMSE" if config.regression else "AUROC"
  mean_metric = float(np.mean(per_model_metric))
  prefix = dataset_name.lower()
  details_path = osp.join(output_dir, f"{prefix}_test_details.json")
  text_path = osp.join(output_dir, f"{prefix}_test_results.txt")
  payload = {
      "dataset_name": dataset_name,
      "metric_name": metric_name,
      "n_models": n_models,
      "fold_num": fold_num,
      "num_folds": args.num_folds,
      "num_splits": num_splits,
      "split_idx": split_idx,
      "per_model_metric": per_model_metric,
      "mean_metric": mean_metric,
      "generated_at": datetime.now().isoformat(timespec="seconds"),
      "model_logdir": model_logdir,
      "resolved_from_run_dir": bool(args.run_dir),
  }
  with open(details_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)
  summary = [
      f"Dataset: {dataset_name}",
      f"Fold: {fold_num}/{args.num_folds}",
      f"Models: {n_models}",
      f"Metric: {metric_name}",
      f"Mean test {metric_name}: {mean_metric:.6f}",
      "Per-model test: " + ", ".join([f"{x:.6f}" for x in per_model_metric]),
  ]
  with open(text_path, "w", encoding="utf-8") as f:
    f.write("\n".join(summary) + "\n")
  print(f"Per-model test {metric_name}: {per_model_metric}")
  print(f"Mean test {metric_name}: {mean_metric:.6f}")
  print(f"Saved: {details_path}")
  print(f"Saved: {text_path}")


if __name__ == "__main__":
  main()
