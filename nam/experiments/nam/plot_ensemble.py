#!/usr/bin/env python
# coding=utf-8
"""Standalone NAM ensemble visualization for trained PyTorch checkpoints."""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys
from datetime import datetime
from typing import Dict, List, Tuple

os.environ.setdefault("KMP_USE_SHM", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def _repo_root() -> str:
  return osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))


def _add_repo_parent_to_path(repo_root: str) -> None:
  parent = osp.dirname(repo_root)
  if parent not in sys.path:
    sys.path.insert(0, parent)


_add_repo_parent_to_path(_repo_root())

from neural_additive_models import data_utils
from neural_additive_models.experiments.nam.train import str2bool
from neural_additive_models.runtime import find_checkpoint_path
from neural_additive_models.runtime import load_checkpoint
from neural_additive_models.runtime import resolve_device
from neural_additive_models.training.metrics import calculate_metric
from neural_additive_models.training.trainer import TrainingConfig
from neural_additive_models.training.trainer import create_model


def _load_training_params(run_dir: str) -> Dict[str, object]:
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
  """Resolve the model directory from a fold-level run directory."""
  run_dir = osp.normpath(run_dir)
  parent = osp.dirname(run_dir)
  if osp.basename(parent) == "training":
    logdir_root = parent
  else:
    logdir_root = osp.join(parent, "training")
  return osp.join(logdir_root, f"fold_{fold_num}", f"split_{split_idx}")


def inverse_min_max_scaler(x, min_val, max_val):
  """Invert the project min-max scaling."""
  return (x + 1) / 2 * (max_val - min_val) + min_val


def load_col_min_max(dataset_name: str) -> Dict[str, Tuple[float, float]]:
  """Load column min/max ranges from the raw dataset."""
  dataset_map = {
      "Housing": data_utils.load_california_housing_data,
      "BreastCancer": data_utils.load_breast_data,
      "Recidivism": data_utils.load_recidivism_data,
      "Fico": data_utils.load_fico_score_data,
      "Credit": data_utils.load_credit_data,
      "Adult": data_utils.load_adult_data,
      "Telco": data_utils.load_telco_churn_data,
  }
  if dataset_name not in dataset_map:
    raise ValueError(f"{dataset_name} not found!")
  dataset = dataset_map[dataset_name]()
  x_df = dataset["X"].copy()
  is_categorical = np.array([dt.kind == "O" for dt in x_df.dtypes])
  categorical_cols = x_df.columns.values[is_categorical].tolist()
  numerical_cols = x_df.columns.values[~is_categorical].tolist()
  if categorical_cols:
    transformed = pd.get_dummies(x_df, columns=categorical_cols, prefix_sep=": ", dtype=np.float32)
  else:
    transformed = x_df.copy()
  if numerical_cols:
    transformed[numerical_cols] = transformed[numerical_cols].astype(np.float32)
  col_min_max = {}
  for col in transformed.columns:
    values = transformed[col].values
    col_min_max[col] = (float(np.min(values)), float(np.max(values)))
  return col_min_max


def compute_all_indices(
    data_x: np.ndarray,
    unique_features: List[np.ndarray],
    column_names: List[str],
) -> Dict[str, np.ndarray]:
  """Map each sample value to the corresponding feature grid index."""
  all_indices = {}
  for index, column_name in enumerate(column_names):
    x_i = data_x[:, index]
    all_indices[column_name] = np.searchsorted(unique_features[index][:, 0], x_i, "left")
  return all_indices


def get_test_predictions(model, x_test: np.ndarray, batch_size: int, device: torch.device):
  """Predict on test data in batches."""
  predictions = []
  model.eval()
  with torch.no_grad():
    for start in range(0, x_test.shape[0], batch_size):
      batch = torch.as_tensor(x_test[start:start + batch_size], dtype=torch.float32, device=device)
      predictions.append(model(batch, training=False).detach().cpu().numpy())
  return np.concatenate(predictions, axis=0)


def get_feature_predictions(model, unique_features: List[np.ndarray], device: torch.device):
  """Predict each feature shape function on its unique-value grid."""
  feature_predictions = []
  model.eval()
  with torch.no_grad():
    for feature_index, feature_values in enumerate(unique_features):
      batch = torch.as_tensor(feature_values, dtype=torch.float32, device=device)
      predictions = model.feature_nns[feature_index](batch, training=False)
      feature_predictions.append(predictions.detach().cpu().numpy().squeeze())
  return feature_predictions


def compute_model_mean_pred(model_hist_data, all_indices, column_names):
  """Compute the feature-wise average contribution per model."""
  return {col: float(np.mean(model_hist_data[col][all_indices[col]])) for col in column_names}


def compute_mean_feature_importance(avg_hist_data, mean_pred):
  """Compute average absolute contributions."""
  mean_abs_score = {}
  for key in avg_hist_data:
    mean_abs_score[key] = np.mean(np.abs(avg_hist_data[key] - mean_pred[key]))
  labels, scores = zip(*mean_abs_score.items())
  return labels, scores


def plot_mean_feature_importance(dataset_name, cols, scores, output_path):
  """Plot aggregated feature importance."""
  fig = plt.figure(figsize=(7, 4.5))
  indices = np.arange(len(cols))
  order = np.argsort(scores)
  ordered_cols = [cols[i] for i in order]
  ordered_scores = [scores[i] for i in order]
  plt.bar(indices, ordered_scores, width=0.6, label="NAM Ensemble")
  plt.xticks(indices, ordered_cols, rotation=75, fontsize=10, ha="right")
  plt.ylabel("Mean Absolute Score", fontsize=12)
  plt.legend(loc="upper right", fontsize=10)
  plt.title(f"Overall Importance: {dataset_name}", fontsize=13)
  plt.tight_layout()
  fig.savefig(output_path, dpi=180)


def shade_by_density_blocks(
    hist_data,
    num_rows,
    num_cols,
    unique_features_original,
    single_features_original,
    categorical_names,
    n_blocks=20,
    color=(0.9, 0.5, 0.5),
):
  """Shade plots according to feature density."""
  hist_data_pairs = sorted(list(hist_data.items()), key=lambda x: x[0])
  min_y = np.min([np.min(pair[1]) for pair in hist_data_pairs])
  max_y = np.max([np.max(pair[1]) for pair in hist_data_pairs])
  span = max_y - min_y
  min_y -= 0.01 * span
  max_y += 0.01 * span
  for index, (name, _) in enumerate(hist_data_pairs):
    unique_x_data = unique_features_original[name]
    single_feature_data = single_features_original[name]
    ax = plt.subplot(num_rows, num_cols, index + 1)
    min_x = np.min(unique_x_data)
    max_x = np.max(unique_x_data)
    block_count = min(n_blocks, len(unique_x_data))
    if name in categorical_names:
      min_x -= 0.5
      max_x += 0.5
    segment_width = (max_x - min_x) / block_count
    density = np.histogram(single_feature_data, bins=block_count)
    normalized_density = density[0] / np.maximum(np.max(density[0]), 1e-12)
    for block_index in range(block_count):
      start_x = min_x + segment_width * block_index
      end_x = min_x + segment_width * (block_index + 1)
      alpha = min(1.0, 0.01 + normalized_density[block_index])
      rect = patches.Rectangle(
          (start_x, min_y - 1),
          end_x - start_x,
          max_y - min_y + 1,
          linewidth=0.01,
          edgecolor=color,
          facecolor=color,
          alpha=alpha,
      )
      ax.add_patch(rect)


def plot_all_hist(
    dataset_name,
    hist_data,
    num_rows,
    num_cols,
    color_base,
    col_names_map,
    categorical_names,
    mean_pred,
    unique_features_original,
    linewidth=3.0,
    min_y=None,
    max_y=None,
    alpha=1.0,
):
  """Plot shape functions for each feature."""
  hist_data_pairs = sorted(list(hist_data.items()), key=lambda x: x[0])
  if min_y is None:
    min_y = np.min([np.min(values) for _, values in hist_data_pairs])
  if max_y is None:
    max_y = np.max([np.max(values) for _, values in hist_data_pairs])
  span = max_y - min_y
  min_y -= 0.01 * span
  max_y += 0.01 * span
  col_mapping = col_names_map[dataset_name]
  for index, (name, pred) in enumerate(hist_data_pairs):
    center = mean_pred[name]
    unique_x_data = unique_features_original[name]
    plt.subplot(num_rows, num_cols, index + 1)
    if name in categorical_names:
      unique_x_data = np.round(unique_x_data, decimals=1)
      step_loc = "mid" if len(unique_x_data) <= 2 else "post"
      unique_plot_data = np.array(unique_x_data) - 0.5
      unique_plot_data[-1] += 1
      plt.step(unique_plot_data, pred - center, color=color_base, linewidth=linewidth, where=step_loc, alpha=alpha)
      plt.xticks(unique_x_data, labels=unique_x_data, fontsize=10)
    else:
      plt.plot(unique_x_data, pred - center, color=color_base, linewidth=linewidth, alpha=alpha)
      plt.xticks(fontsize=10)
    plt.ylim(min_y, max_y)
    plt.yticks(fontsize=10)
    min_x = np.min(unique_x_data)
    max_x = np.max(unique_x_data)
    if name in categorical_names:
      min_x -= 0.5
      max_x += 0.5
    plt.xlim(min_x, max_x)
    if index % num_cols == 0:
      plt.ylabel("Feature Contribution", fontsize=11)
    plt.xlabel(col_mapping[name], fontsize=11)
  return min_y, max_y


def build_parser() -> argparse.ArgumentParser:
  """Build the plot CLI parser."""
  repo_root = _repo_root()
  parser = argparse.ArgumentParser(description="Plot NAM ensemble shape functions.")
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
  parser.add_argument("--run_test_metrics", type=str2bool, nargs="?", const=True, default=False)
  parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"])
  parser.add_argument("--output_dir", default=None)
  return parser


def main(argv=None):
  """CLI entrypoint."""
  args = build_parser().parse_args(argv)
  repo_root = _repo_root()
  default_model_logdir = osp.join(repo_root, "outputs", "nam", "training", "fold_5", "split_1")
  default_output_dir = osp.join(repo_root, "outputs", "nam", "plots", "housing_example", "fold_5")
  cfg = _load_training_params(args.run_dir) if args.run_dir else {}
  merged = {**cfg}
  for key, value in vars(args).items():
    if value is not None:
      merged[key] = value
  dataset_name = merged.get("dataset_name", "Housing")
  n_models = int(merged.get("n_models", 20))
  fold_num = int(merged.get("fold_num", 5))
  num_splits = int(merged.get("num_splits", 3))
  split_idx = int(merged.get("split_idx", 1))
  model_logdir = args.model_logdir or (
      _infer_model_logdir(args.run_dir, fold_num, split_idx) if args.run_dir else default_model_logdir)
  output_dir = args.output_dir or (
      osp.join(args.run_dir, "visualization_outputs") if args.run_dir else default_output_dir)
  os.makedirs(output_dir, exist_ok=True)
  device = resolve_device(args.device)
  is_regression = (
      args.regression if args.regression is not None else merged.get("regression", dataset_name in ["Housing", "Fico"]))
  config = TrainingConfig(
      training_epochs=1,
      dropout=float(merged.get("dropout", 0.0)),
      feature_dropout=float(merged.get("feature_dropout", 0.0)),
      num_basis_functions=int(merged.get("num_basis_functions", 64)),
      units_multiplier=int(merged.get("units_multiplier", 2)),
      activation=str(merged.get("activation", "relu")),
      regression=is_regression,
      shallow=bool(merged.get("shallow", False)),
      use_dnn=bool(merged.get("use_dnn", False)),
      max_checkpoints_to_keep=1,
      device=args.device,
  )

  data_x, data_y, column_names = data_utils.load_dataset(dataset_name)
  col_min_max = load_col_min_max(dataset_name)
  (x_train_all, y_train_all), test_dataset = data_utils.get_train_test_fold(
      data_x,
      data_y,
      fold_num=fold_num,
      num_folds=args.num_folds,
      stratified=not is_regression,
  )
  data_gen = data_utils.split_training_dataset(
      x_train_all,
      y_train_all,
      n_splits=num_splits,
      stratified=not is_regression,
  )
  split = None
  for _ in range(split_idx):
    split = next(data_gen)
  (x_train, _), _ = split

  num_features = data_x.shape[1]
  single_features = np.split(data_x, num_features, axis=1)
  unique_features = [np.unique(x, axis=0) for x in single_features]

  unique_features_original = {}
  single_features_original = {}
  for index, col in enumerate(column_names):
    min_val, max_val = col_min_max[col]
    unique_features_original[col] = inverse_min_max_scaler(unique_features[index][:, 0], min_val, max_val)
    single_features_original[col] = inverse_min_max_scaler(single_features[index][:, 0], min_val, max_val)
  all_indices = compute_all_indices(data_x, unique_features, column_names)

  col_names_map = {
      "Housing": {
          "MedInc": "Median Income",
          "HouseAge": "Median House Age",
          "AveRooms": "# Avg Rooms",
          "AveBedrms": "# Avg Bedrooms",
          "Population": "Block Population",
          "AveOccup": "# Avg Occupancy",
          "Latitude": "Latitude",
          "Longitude": "Longitude",
      }
  }
  if dataset_name not in col_names_map:
    col_names_map[dataset_name] = {name: name for name in column_names}
  categorical_names = []

  per_model_hist_data = []
  per_model_mean_pred = []
  per_model_test_metric = []
  for model_idx in range(n_models):
    checkpoint_path = find_checkpoint_path(osp.join(model_logdir, f"model_{model_idx}"))
    checkpoint = load_checkpoint(checkpoint_path, map_location=device)
    model = create_model(config, x_train).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    if args.run_test_metrics:
      test_predictions = get_test_predictions(model, test_dataset[0], batch_size=256, device=device)
      per_model_test_metric.append(float(calculate_metric(test_dataset[1], test_predictions, regression=is_regression)))
    feature_predictions = get_feature_predictions(model, unique_features, device=device)
    model_hist_data = {col: pred for col, pred in zip(column_names, feature_predictions)}
    per_model_hist_data.append(model_hist_data)
    per_model_mean_pred.append(compute_model_mean_pred(model_hist_data, all_indices, column_names))

  avg_hist_data = {
      col: np.mean(np.stack([model_hist[col] for model_hist in per_model_hist_data], axis=0), axis=0)
      for col in column_names
  }
  mean_pred_ensemble = compute_model_mean_pred(avg_hist_data, all_indices, column_names)
  x1, x2 = compute_mean_feature_importance(avg_hist_data, mean_pred_ensemble)
  cols = [col_names_map[dataset_name][x] for x in x1]

  prefix = dataset_name.lower()
  importance_path = osp.join(output_dir, f"{prefix}_feature_importance.png")
  shape_path = osp.join(output_dir, f"{prefix}_shape_plots_ensemble.png")
  details_path = osp.join(output_dir, f"{prefix}_ensemble_plot_run_details.json")
  text_summary_path = osp.join(output_dir, f"{prefix}_test_results.txt")
  plot_mean_feature_importance(dataset_name, cols, x2, importance_path)

  num_cols = 4
  num_rows = int(np.ceil(num_features / num_cols))
  fig = plt.figure(figsize=(num_cols * 4.5, num_rows * 4.5), facecolor="w", edgecolor="k")
  min_y, max_y = plot_all_hist(
      dataset_name=dataset_name,
      hist_data=avg_hist_data,
      num_rows=num_rows,
      num_cols=num_cols,
      color_base=[0.2, 0.35, 0.9],
      col_names_map=col_names_map,
      categorical_names=categorical_names,
      mean_pred=mean_pred_ensemble,
      unique_features_original=unique_features_original,
      linewidth=3.0,
      alpha=1.0,
  )
  for model_hist, model_mean in zip(per_model_hist_data, per_model_mean_pred):
    plot_all_hist(
        dataset_name=dataset_name,
        hist_data=model_hist,
        num_rows=num_rows,
        num_cols=num_cols,
        color_base=[0.5, 0.5, 0.5],
        col_names_map=col_names_map,
        categorical_names=categorical_names,
        mean_pred=model_mean,
        unique_features_original=unique_features_original,
        linewidth=1.0,
        min_y=min_y,
        max_y=max_y,
        alpha=0.3,
    )
  shade_by_density_blocks(
      hist_data=avg_hist_data,
      num_rows=num_rows,
      num_cols=num_cols,
      unique_features_original=unique_features_original,
      single_features_original=single_features_original,
      categorical_names=categorical_names,
  )
  plt.tight_layout()
  fig.savefig(shape_path, dpi=180)

  metric_name = "RMSE" if is_regression else "AUROC"
  payload = {
      "dataset_name": dataset_name,
      "metric_name": metric_name,
      "n_models": n_models,
      "fold_num": fold_num,
      "num_folds": args.num_folds,
      "num_splits": num_splits,
      "split_idx": split_idx,
      "run_test_metrics": args.run_test_metrics,
      "per_model_test_metric": per_model_test_metric,
      "generated_at": datetime.now().isoformat(timespec="seconds"),
      "model_logdir": model_logdir,
      "feature_importance_path": importance_path,
      "shape_plot_path": shape_path,
  }
  with open(details_path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)
  if per_model_test_metric:
    with open(text_summary_path, "w", encoding="utf-8") as f:
      f.write(f"Mean test {metric_name}: {float(np.mean(per_model_test_metric)):.6f}\n")
      f.write("Per-model test: " + ", ".join([f"{value:.6f}" for value in per_model_test_metric]) + "\n")
  print(f"Saved: {importance_path}")
  print(f"Saved: {shape_path}")
  print(f"Saved: {details_path}")


if __name__ == "__main__":
  main()
