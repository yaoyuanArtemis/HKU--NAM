#!/usr/bin/env python
# coding=utf-8
"""COMPAS single-task and multitask NAM experiment runner."""

from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Sequence

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

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import WeightedRandomSampler
from tqdm.auto import tqdm

from neural_additive_models import data_utils
from neural_additive_models import models
from neural_additive_models.experiments.nam.train import str2bool
from neural_additive_models.runtime import resolve_device
from neural_additive_models.runtime import save_checkpoint
from neural_additive_models.training.data import create_eval_loader
from neural_additive_models.training.data import create_train_loader
from neural_additive_models.training.trainer import infer_num_units


DISPLAY_TASK_NAMES = ("Women", "Men")
TASK_VALUE_TO_INDEX = {"Female": 0, "Male": 1}
COMPAS_FEATURE_COLUMNS = [
    "age",
    "juv_fel_count",
    "juv_misd_count",
    "juv_other_count",
    "priors_count",
    "c_charge_degree",
    "race",
    "sex",
]
FEATURE_LABELS = {
    "age": "Age",
    "juv_fel_count": "Juv. felonies",
    "juv_misd_count": "Juv. misdemeanors",
    "juv_other_count": "Juv. other",
    "priors_count": "Prior Counts",
    "c_charge_degree": "Charge Degree",
    "race": "Race",
    "sex": "Gender",
}
PINK = (0.95, 0.67, 0.75)
BLUE = (0.18, 0.36, 0.88)
GREEN = (0.10, 0.62, 0.36)
RED = (0.86, 0.36, 0.38)

SINGLE_TASK_PLOT_ORDER = ["age", "c_charge_degree", "priors_count", "race", "sex"]
MULTITASK_PLOT_ORDER = ["c_charge_degree", "race"]


@dataclass
class ExperimentConfig:
  """Hyperparameters for COMPAS experiments."""

  training_epochs: int
  learning_rate: float
  batch_size: int
  decay_rate: float
  dropout: float
  feature_dropout: float
  num_basis_functions: int
  units_multiplier: int
  activation: str
  shallow: bool
  output_regularization: float
  l2_regularization: float
  early_stopping_epochs: int
  n_models: int
  n_folds: int
  seed: int
  device: str
  output_dir: str
  figure_fold: int
  mode: str
  include_sex_feature_for_multitask: bool


class MultiTaskTabularDataset(Dataset):
  """Dense tabular dataset with task masks for multitask classification."""

  def __init__(self, features: np.ndarray, targets: np.ndarray, masks: np.ndarray) -> None:
    self.features = torch.as_tensor(features, dtype=torch.float32)
    self.targets = torch.as_tensor(targets, dtype=torch.float32)
    self.masks = torch.as_tensor(masks, dtype=torch.float32)

  def __len__(self) -> int:
    return int(self.features.shape[0])

  def __getitem__(self, index: int):
    return self.features[index], self.targets[index], self.masks[index]


def build_parser() -> argparse.ArgumentParser:
  """Create the CLI parser."""
  parser = argparse.ArgumentParser(description="Run COMPAS single-task and multitask NAM experiments.")
  parser.add_argument("--mode", choices=["all", "cv", "figure"], default="all")
  parser.add_argument("--training_epochs", type=int, default=80)
  parser.add_argument("--learning_rate", type=float, default=0.02082)
  parser.add_argument("--batch_size", type=int, default=512)
  parser.add_argument("--decay_rate", type=float, default=0.995)
  parser.add_argument("--dropout", type=float, default=0.1)
  parser.add_argument("--feature_dropout", type=float, default=0.05)
  parser.add_argument("--num_basis_functions", type=int, default=64)
  parser.add_argument("--units_multiplier", type=int, default=2)
  parser.add_argument("--activation", type=str, default="relu")
  parser.add_argument("--shallow", type=str2bool, nargs="?", const=True, default=False)
  parser.add_argument("--output_regularization", type=float, default=0.2078)
  parser.add_argument("--l2_regularization", type=float, default=0.0)
  parser.add_argument("--early_stopping_epochs", type=int, default=20)
  parser.add_argument("--n_models", type=int, default=10)
  parser.add_argument("--n_folds", type=int, default=5)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--figure_fold", type=int, default=1)
  parser.add_argument("--include_sex_feature_for_multitask", type=str2bool, nargs="?", const=True, default=True)
  parser.add_argument("--device", type=str, default="cpu", choices=["auto", "cuda", "mps", "cpu"])
  parser.add_argument("--output_dir", type=str, default=osp.join(_repo_root(), "outputs", "compas_multitask"))
  return parser


def sigmoid(values: np.ndarray) -> np.ndarray:
  """Apply a numerically stable sigmoid to logits."""
  values = np.asarray(values, dtype=np.float64)
  result = np.empty_like(values, dtype=np.float64)
  positive = values >= 0
  result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
  exp_values = np.exp(values[~positive])
  result[~positive] = exp_values / (1.0 + exp_values)
  return result


def make_experiment_config(args: argparse.Namespace) -> ExperimentConfig:
  """Convert CLI args into a typed config."""
  return ExperimentConfig(
      training_epochs=args.training_epochs,
      learning_rate=args.learning_rate,
      batch_size=args.batch_size,
      decay_rate=args.decay_rate,
      dropout=args.dropout,
      feature_dropout=args.feature_dropout,
      num_basis_functions=args.num_basis_functions,
      units_multiplier=args.units_multiplier,
      activation=args.activation,
      shallow=args.shallow,
      output_regularization=args.output_regularization,
      l2_regularization=args.l2_regularization,
      early_stopping_epochs=args.early_stopping_epochs,
      n_models=args.n_models,
      n_folds=args.n_folds,
      seed=args.seed,
      device=args.device,
      output_dir=args.output_dir,
      figure_fold=args.figure_fold,
      mode=args.mode,
      include_sex_feature_for_multitask=args.include_sex_feature_for_multitask,
  )


def ensure_dir(path: str) -> str:
  """Create a directory if needed and return it."""
  os.makedirs(path, exist_ok=True)
  return path


def set_torch_seed(seed: int) -> None:
  """Seed numpy and torch RNGs."""
  np.random.seed(seed)
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)


def feature_output_regularization(model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
  """Penalize the mean squared feature contributions."""
  per_feature_outputs = model.calc_outputs(inputs, training=False)
  penalties = [torch.mean(outputs.square()) for outputs in per_feature_outputs]
  return torch.stack(penalties).mean()


def weight_decay(model: nn.Module, num_networks: int) -> torch.Tensor:
  """Compute L2 regularization across trainable parameters."""
  penalties = [0.5 * parameter.square().sum() for parameter in model.parameters() if parameter.requires_grad]
  if not penalties:
    return torch.tensor(0.0)
  return torch.stack(penalties).sum() / max(num_networks, 1)


def build_group_labels(labels: np.ndarray, task_index: np.ndarray) -> np.ndarray:
  """Create stratification labels that preserve both sex and recidivism rate."""
  return np.array([f"{int(task)}_{int(label)}" for task, label in zip(task_index, labels)], dtype=object)


def prepare_single_task_compas() -> Dict[str, object]:
  """Load COMPAS data for the single-task experiment."""
  df = data_utils.load_recidivism_dataframe()
  x_df = df[COMPAS_FEATURE_COLUMNS].copy()
  features, column_names = data_utils.transform_data(x_df)
  return {
      "features": features.astype(np.float32),
      "labels": df["two_year_recid"].to_numpy(dtype=np.float32),
      "task_index": np.array([TASK_VALUE_TO_INDEX[value] for value in df["sex"].to_numpy()], dtype=np.int64),
      "column_names": column_names,
      "raw_frame": df,
      "raw_features": x_df,
      "feature_columns": list(COMPAS_FEATURE_COLUMNS),
  }


def prepare_multitask_compas(include_sex_feature: bool) -> Dict[str, object]:
  """Load COMPAS data for the multitask experiment."""
  dataset = data_utils.load_recidivism_multitask_data(include_sex_feature=include_sex_feature)
  return {
      "features": dataset["X"],
      "targets": dataset["y"],
      "masks": dataset["mask"],
      "task_index": dataset["task_index"],
      "column_names": dataset["column_names"],
      "raw_frame": dataset["raw_frame"],
      "feature_columns": dataset["feature_columns"],
  }


def build_single_task_model(x_train: np.ndarray, config: ExperimentConfig) -> models.NAM:
  """Create a single-task NAM."""
  num_units = infer_num_units(
      x_train=x_train,
      num_basis_functions=config.num_basis_functions,
      units_multiplier=config.units_multiplier,
  )
  return models.NAM(
      num_inputs=x_train.shape[1],
      num_units=num_units,
      shallow=config.shallow,
      feature_dropout=config.feature_dropout,
      dropout=config.dropout,
      activation=config.activation,
  )


def build_multitask_model(x_train: np.ndarray, config: ExperimentConfig, num_tasks: int) -> models.MultiTaskNAM:
  """Create a multitask NAM."""
  num_units = infer_num_units(
      x_train=x_train,
      num_basis_functions=config.num_basis_functions,
      units_multiplier=config.units_multiplier,
  )
  return models.MultiTaskNAM(
      num_inputs=x_train.shape[1],
      num_units=num_units,
      num_tasks=num_tasks,
      shallow=config.shallow,
      feature_dropout=config.feature_dropout,
      dropout=config.dropout,
      activation=config.activation,
  )


def masked_bce_with_logits(logits: torch.Tensor, targets: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
  """Binary cross entropy with task masking."""
  loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
  return (loss * masks).sum() / masks.sum().clamp_min(1.0)


def create_multitask_train_loader(
    features: np.ndarray,
    targets: np.ndarray,
    masks: np.ndarray,
    task_index: np.ndarray,
    batch_size: int,
) -> DataLoader:
  """Create a balanced train loader across task/label combinations."""
  dataset = MultiTaskTabularDataset(features, targets, masks)
  labels = targets[np.arange(targets.shape[0]), task_index].astype(np.int64)
  group_labels = task_index.astype(np.int64) * 2 + labels
  group_counts = np.bincount(group_labels, minlength=4)
  group_weights = np.zeros_like(group_counts, dtype=np.float64)
  non_zero = group_counts > 0
  group_weights[non_zero] = 1.0 / group_counts[non_zero]
  sample_weights = group_weights[group_labels]
  sampler = WeightedRandomSampler(
      weights=torch.as_tensor(sample_weights, dtype=torch.double),
      num_samples=len(sample_weights),
      replacement=True,
  )
  return DataLoader(dataset, batch_size=min(batch_size, len(dataset)), sampler=sampler)


def predict_single_model(
    model: nn.Module,
    features: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
  """Predict single-task logits in batches."""
  loader = create_eval_loader(features, batch_size=batch_size)
  outputs = []
  model.eval()
  with torch.no_grad():
    for batch_features, _ in loader:
      outputs.append(model(batch_features.to(device), training=False).detach().cpu().numpy())
  return np.concatenate(outputs, axis=0)


def predict_multitask_model(
    model: nn.Module,
    features: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
  """Predict multitask logits in batches."""
  loader = create_eval_loader(features, batch_size=batch_size)
  outputs = []
  model.eval()
  with torch.no_grad():
    for batch_features, _ in loader:
      outputs.append(model(batch_features.to(device), training=False).detach().cpu().numpy())
  return np.concatenate(outputs, axis=0)


def compute_single_task_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    task_index: np.ndarray,
) -> Dict[str, float]:
  """Compute Women, Men and Combined AUROC for single-task predictions."""
  women_mask = task_index == 0
  men_mask = task_index == 1
  return {
      "women_auc": float(roc_auc_score(labels[women_mask], probabilities[women_mask])),
      "men_auc": float(roc_auc_score(labels[men_mask], probabilities[men_mask])),
      "combined_auc": float(roc_auc_score(labels, probabilities)),
  }


def compute_multitask_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    task_index: np.ndarray,
) -> Dict[str, float]:
  """Compute Women, Men and Combined AUROC for multitask predictions."""
  women_mask = task_index == 0
  men_mask = task_index == 1
  matched_probabilities = probabilities[np.arange(probabilities.shape[0]), task_index]
  return {
      "women_auc": float(roc_auc_score(labels[women_mask], probabilities[women_mask, 0])),
      "men_auc": float(roc_auc_score(labels[men_mask], probabilities[men_mask, 1])),
      "combined_auc": float(roc_auc_score(labels, matched_probabilities)),
  }


def _checkpoint_payload(model: nn.Module, epoch: int, metric_value: float) -> Dict[str, object]:
  """Build a minimal checkpoint payload."""
  return {
      "epoch": epoch,
      "metric_value": metric_value,
      "state_dict": model.state_dict(),
      "model_type": model.__class__.__name__,
  }


def train_single_task_model(
    model_index: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    train_task_index: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    val_task_index: np.ndarray,
    config: ExperimentConfig,
    device: torch.device,
    model_dir: str,
):
  """Train one single-task NAM and return the best checkpointed model."""
  del train_task_index
  set_torch_seed(config.seed + model_index)
  model = build_single_task_model(x_train, config).to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
  scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.decay_rate)
  train_loader = create_train_loader(
      x_train=x_train,
      y_train=y_train,
      batch_size=min(config.batch_size, x_train.shape[0]),
      regression=False,
  )

  best_metric = -np.inf
  best_epoch = 0
  best_state = None
  patience = 0
  best_dir = ensure_dir(osp.join(model_dir, "best_checkpoint"))

  for epoch in range(1, config.training_epochs + 1):
    model.train()
    for batch_features, batch_targets in train_loader:
      batch_features = batch_features.to(device)
      batch_targets = batch_targets.to(device)
      optimizer.zero_grad()
      logits = model(batch_features, training=True)
      loss = F.binary_cross_entropy_with_logits(logits, batch_targets)
      if config.output_regularization > 0:
        loss = loss + config.output_regularization * feature_output_regularization(model, batch_features)
      if config.l2_regularization > 0:
        loss = loss + config.l2_regularization * weight_decay(model, num_networks=len(model.feature_nns))
      loss.backward()
      optimizer.step()
    scheduler.step()

    val_probabilities = sigmoid(predict_single_model(model, x_val, config.batch_size, device))
    validation_metric = compute_single_task_metrics(y_val, val_probabilities, val_task_index)["combined_auc"]
    if validation_metric > best_metric:
      best_metric = validation_metric
      best_epoch = epoch
      patience = 0
      best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
      save_checkpoint(_checkpoint_payload(model, epoch, validation_metric), osp.join(best_dir, "model.pt"))
    else:
      patience += 1
      if patience >= config.early_stopping_epochs:
        break

  if best_state is None:
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    save_checkpoint(_checkpoint_payload(model, config.training_epochs, best_metric), osp.join(best_dir, "model.pt"))
  model.load_state_dict(best_state)
  model.eval()
  return model, {"best_epoch": best_epoch or config.training_epochs, "best_validation_auc": float(best_metric)}


def train_multitask_model(
    model_index: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    train_masks: np.ndarray,
    train_task_index: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    val_task_index: np.ndarray,
    config: ExperimentConfig,
    device: torch.device,
    model_dir: str,
):
  """Train one multitask NAM and return the best checkpointed model."""
  set_torch_seed(config.seed + model_index)
  model = build_multitask_model(x_train, config, num_tasks=y_train.shape[1]).to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
  scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.decay_rate)
  train_loader = create_multitask_train_loader(
      features=x_train,
      targets=y_train,
      masks=train_masks,
      task_index=train_task_index,
      batch_size=config.batch_size,
  )

  best_metric = -np.inf
  best_epoch = 0
  best_state = None
  patience = 0
  best_dir = ensure_dir(osp.join(model_dir, "best_checkpoint"))

  for epoch in range(1, config.training_epochs + 1):
    model.train()
    for batch_features, batch_targets, batch_masks in train_loader:
      batch_features = batch_features.to(device)
      batch_targets = batch_targets.to(device)
      batch_masks = batch_masks.to(device)
      optimizer.zero_grad()
      logits = model(batch_features, training=True)
      loss = masked_bce_with_logits(logits, batch_targets, batch_masks)
      if config.output_regularization > 0:
        loss = loss + config.output_regularization * feature_output_regularization(model, batch_features)
      if config.l2_regularization > 0:
        loss = loss + config.l2_regularization * weight_decay(model, num_networks=len(model.feature_nns))
      loss.backward()
      optimizer.step()
    scheduler.step()

    val_probabilities = sigmoid(predict_multitask_model(model, x_val, config.batch_size, device))
    validation_metric = compute_multitask_metrics(
        labels=y_val[np.arange(y_val.shape[0]), val_task_index],
        probabilities=val_probabilities,
        task_index=val_task_index,
    )["combined_auc"]
    if validation_metric > best_metric:
      best_metric = validation_metric
      best_epoch = epoch
      patience = 0
      best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
      save_checkpoint(_checkpoint_payload(model, epoch, validation_metric), osp.join(best_dir, "model.pt"))
    else:
      patience += 1
      if patience >= config.early_stopping_epochs:
        break

  if best_state is None:
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    save_checkpoint(_checkpoint_payload(model, config.training_epochs, best_metric), osp.join(best_dir, "model.pt"))
  model.load_state_dict(best_state)
  model.eval()
  return model, {"best_epoch": best_epoch or config.training_epochs, "best_validation_auc": float(best_metric)}


def ensemble_mean(predictions: Sequence[np.ndarray]) -> np.ndarray:
  """Average a list of per-model probability arrays."""
  return np.mean(np.stack(predictions, axis=0), axis=0)


def summarize_metric_list(values: Iterable[float]) -> Dict[str, float]:
  """Return mean and std summary for a list of metrics."""
  array = np.asarray(list(values), dtype=np.float64)
  return {"mean": float(np.mean(array)), "std": float(np.std(array, ddof=1) if len(array) > 1 else 0.0)}


def run_cross_validation(
    single_data: Dict[str, object],
    multitask_data: Dict[str, object],
    config: ExperimentConfig,
    device: torch.device,
) -> Dict[str, object]:
  """Run 5-fold COMPAS experiments for single-task and multitask NAMs."""
  labels = np.asarray(single_data["labels"], dtype=np.float32)
  task_index = np.asarray(single_data["task_index"], dtype=np.int64)
  strata = build_group_labels(labels, task_index)
  splitter = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=config.seed)
  cv_results = {"single_task": [], "multitask": []}

  for fold_index, (train_val_index, test_index) in enumerate(splitter.split(single_data["features"], strata), start=1):
    fold_dir = ensure_dir(osp.join(config.output_dir, "training", f"fold_{fold_index}"))
    train_val_strata = strata[train_val_index]
    validation_splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=0.125,
        random_state=config.seed + fold_index,
    )
    train_rel_index, val_rel_index = next(validation_splitter.split(single_data["features"][train_val_index], train_val_strata))
    train_index = train_val_index[train_rel_index]
    val_index = train_val_index[val_rel_index]

    single_model_probabilities = []
    multitask_model_probabilities = []
    single_metadata = []
    multitask_metadata = []

    single_fold_dir = ensure_dir(osp.join(fold_dir, "single_task"))
    multitask_fold_dir = ensure_dir(osp.join(fold_dir, "multitask"))

    for model_index in tqdm(range(config.n_models), desc=f"Fold {fold_index} models"):
      single_model, single_info = train_single_task_model(
          model_index=model_index,
          x_train=single_data["features"][train_index],
          y_train=labels[train_index],
          train_task_index=task_index[train_index],
          x_val=single_data["features"][val_index],
          y_val=labels[val_index],
          val_task_index=task_index[val_index],
          config=config,
          device=device,
          model_dir=ensure_dir(osp.join(single_fold_dir, f"model_{model_index}")),
      )
      single_probs = sigmoid(
          predict_single_model(single_model, single_data["features"][test_index], config.batch_size, device)
      )
      single_model_probabilities.append(single_probs)
      single_metadata.append(single_info)

      multitask_model, multitask_info = train_multitask_model(
          model_index=model_index,
          x_train=multitask_data["features"][train_index],
          y_train=multitask_data["targets"][train_index],
          train_masks=multitask_data["masks"][train_index],
          train_task_index=task_index[train_index],
          x_val=multitask_data["features"][val_index],
          y_val=multitask_data["targets"][val_index],
          val_task_index=task_index[val_index],
          config=config,
          device=device,
          model_dir=ensure_dir(osp.join(multitask_fold_dir, f"model_{model_index}")),
      )
      multitask_probs = sigmoid(
          predict_multitask_model(multitask_model, multitask_data["features"][test_index], config.batch_size, device)
      )
      multitask_model_probabilities.append(multitask_probs)
      multitask_metadata.append(multitask_info)

    single_metrics = compute_single_task_metrics(
        labels=labels[test_index],
        probabilities=ensemble_mean(single_model_probabilities),
        task_index=task_index[test_index],
    )
    multitask_metrics = compute_multitask_metrics(
        labels=labels[test_index],
        probabilities=ensemble_mean(multitask_model_probabilities),
        task_index=task_index[test_index],
    )
    cv_results["single_task"].append({"fold": fold_index, **single_metrics, "models": single_metadata})
    cv_results["multitask"].append({"fold": fold_index, **multitask_metrics, "models": multitask_metadata})

  summary = {
      "single_task": {
          metric_name: summarize_metric_list([fold[metric_name] for fold in cv_results["single_task"]])
          for metric_name in ("women_auc", "men_auc", "combined_auc")
      },
      "multitask": {
          metric_name: summarize_metric_list([fold[metric_name] for fold in cv_results["multitask"]])
          for metric_name in ("women_auc", "men_auc", "combined_auc")
      },
  }
  return {"folds": cv_results, "summary": summary}


def build_feature_groups(
    column_names: Sequence[str],
    raw_features: pd.DataFrame,
    feature_columns: Sequence[str],
) -> Dict[str, Dict[str, object]]:
  """Map original COMPAS features to transformed column groups."""
  groups = {}
  categorical_features = {column for column in feature_columns if raw_features[column].dtype.kind == "O"}
  for feature_name in feature_columns:
    if feature_name in categorical_features:
      group_columns = [name for name in column_names if name.startswith(f"{feature_name}: ")]
      categories = [name.split(": ", 1)[1] for name in group_columns]
      groups[feature_name] = {
          "type": "categorical",
          "columns": group_columns,
          "indices": [column_names.index(name) for name in group_columns],
          "categories": categories,
      }
    else:
      groups[feature_name] = {
          "type": "numeric",
          "columns": [feature_name],
          "indices": [column_names.index(feature_name)],
      }
  return groups


def compute_group_mean_contribution(
    model: nn.Module,
    x_values: np.ndarray,
    group_indices: Sequence[int],
    device: torch.device,
) -> np.ndarray:
  """Compute the mean group contribution on real data."""
  contributions = []
  with torch.no_grad():
    for feature_index in group_indices:
      batch = torch.as_tensor(x_values[:, feature_index:feature_index + 1], dtype=torch.float32, device=device)
      contribution = model.feature_nns[feature_index](batch, training=False).detach().cpu().numpy()
      contributions.append(np.atleast_2d(contribution).reshape(len(x_values), -1))
  total = np.sum(np.stack(contributions, axis=0), axis=0)
  return total.mean(axis=0)


def compute_numeric_curve(
    model: nn.Module,
    x_values: np.ndarray,
    feature_index: int,
    device: torch.device,
) -> Dict[str, np.ndarray]:
  """Compute a mean-centered shape curve for a numeric feature."""
  scaled_values = np.unique(x_values[:, feature_index]).astype(np.float32)
  with torch.no_grad():
    grid = torch.as_tensor(scaled_values[:, None], dtype=torch.float32, device=device)
    predictions = model.feature_nns[feature_index](grid, training=False).detach().cpu().numpy()
  predictions = np.atleast_2d(predictions).reshape(len(scaled_values), -1)
  center = compute_group_mean_contribution(model, x_values, [feature_index], device)
  return {"scaled_grid": scaled_values, "curve": predictions - center}


def compute_categorical_curve(
    model: nn.Module,
    x_values: np.ndarray,
    group_indices: Sequence[int],
    device: torch.device,
) -> Dict[str, np.ndarray]:
  """Compute a mean-centered grouped shape curve for one-hot categorical features."""
  category_values = []
  with torch.no_grad():
    for category_position, _ in enumerate(group_indices):
      total_contribution = None
      for group_position, feature_index in enumerate(group_indices):
        value = np.array([[1.0 if group_position == category_position else 0.0]], dtype=np.float32)
        batch = torch.as_tensor(value, dtype=torch.float32, device=device)
        contribution = model.feature_nns[feature_index](batch, training=False).detach().cpu().numpy().reshape(1, -1)
        total_contribution = contribution if total_contribution is None else total_contribution + contribution
      category_values.append(total_contribution[0])
  curve = np.stack(category_values, axis=0)
  center = compute_group_mean_contribution(model, x_values, group_indices, device)
  return {"curve": curve - center}


def _set_axis_spines(ax: plt.Axes) -> None:
  """Apply the shared axes styling."""
  for spine in ax.spines.values():
    spine.set_linewidth(1.1)
    spine.set_color("#333333")


def _normalize_category_labels(categories: Sequence[str]) -> List[str]:
  """Map raw COMPAS category values to publication-friendly labels."""
  mapping = {
      "African-American": "African\nAmerican",
      "Native American": "Native\nAmerican",
      "M": "Misd.",
      "F": "Fel.",
      "Female": "Female",
      "Male": "Male",
  }
  return [mapping.get(category, category) for category in categories]


def _add_density_background(
    ax: plt.Axes,
    feature_type: str,
    raw_values: Sequence[object],
    categories: Sequence[str] | None = None,
) -> None:
  """Shade each panel using feature density, similar to the NAM ensemble plots."""
  y_min, y_max = ax.get_ylim()
  if feature_type == "numeric":
    raw_array = np.asarray(raw_values, dtype=np.float32)
    unique_values = np.unique(raw_array)
    block_count = int(min(max(len(unique_values), 6), 24))
    counts, edges = np.histogram(raw_array, bins=block_count)
    normalized = counts / max(np.max(counts), 1)
    for left, right, density in zip(edges[:-1], edges[1:], normalized):
      alpha = 0.08 + 0.52 * float(density)
      rect = mpatches.Rectangle(
          (float(left), y_min),
          float(right - left),
          y_max - y_min,
          linewidth=0.7,
          edgecolor=PINK,
          facecolor=PINK,
          alpha=alpha,
          zorder=0,
      )
      ax.add_patch(rect)
  else:
    counts = pd.Series(raw_values).value_counts()
    ordered_counts = np.array([counts.get(category, 0) for category in categories], dtype=np.float32)
    normalized = ordered_counts / max(np.max(ordered_counts), 1.0)
    for index, density in enumerate(normalized):
      rect = mpatches.Rectangle(
          (index - 0.5, y_min),
          1.0,
          y_max - y_min,
          linewidth=0.7,
          edgecolor=PINK,
          facecolor=PINK,
          alpha=0.08 + 0.52 * float(density),
          zorder=0,
      )
      ax.add_patch(rect)
  ax.set_ylim(y_min, y_max)


def _plot_categorical_step(
    ax: plt.Axes,
    x_positions: np.ndarray,
    values: np.ndarray,
    color,
    linewidth: float,
    alpha: float,
    label: str | None = None,
) -> None:
  """Render grouped categorical contributions as a step curve."""
  step_x = np.concatenate(([x_positions[0] - 0.5], x_positions + 0.5))
  step_y = np.concatenate((values, [values[-1]]))
  ax.step(step_x, step_y, where="post", color=color, linewidth=linewidth, alpha=alpha, label=label, zorder=3)


def _style_panel(
    ax: plt.Axes,
    title: str,
    xlabel: str,
    ylim: Sequence[float],
    show_ylabel: bool,
) -> None:
  """Apply common labels and tick styling."""
  ax.set_title(title, fontsize=15, pad=10)
  ax.set_xlabel(xlabel, fontsize=12)
  if show_ylabel:
    ax.set_ylabel("Recidivism Risk", fontsize=12)
  ax.set_ylim(*ylim)
  ax.tick_params(axis="both", labelsize=11, length=0)
  _set_axis_spines(ax)
  ax.grid(False)


def _aggregate_single_task_curves(
    single_models: Sequence[nn.Module],
    single_data: Dict[str, object],
    single_groups: Dict[str, Dict[str, object]],
    feature_name: str,
    device: torch.device,
) -> Dict[str, object]:
  """Collect per-model and mean curves for one single-task feature."""
  group = single_groups[feature_name]
  raw_values = single_data["raw_features"][feature_name].to_numpy()
  if group["type"] == "numeric":
    x_grid = None
    curves = []
    for model in single_models:
      result = compute_numeric_curve(model, single_data["features"], group["indices"][0], device)
      x_grid = inverse_numeric_axis(raw_values, result["scaled_grid"])
      curves.append(result["curve"][:, 0])
    curves_array = np.stack(curves, axis=0)
    return {"type": "numeric", "x": x_grid, "curves": curves_array, "mean": np.mean(curves_array, axis=0)}

  x_positions = np.arange(len(group["categories"]), dtype=np.float32)
  curves = []
  for model in single_models:
    result = compute_categorical_curve(model, single_data["features"], group["indices"], device)
    curves.append(result["curve"][:, 0])
  curves_array = np.stack(curves, axis=0)
  return {
      "type": "categorical",
      "x": x_positions,
      "curves": curves_array,
      "mean": np.mean(curves_array, axis=0),
      "categories": group["categories"],
  }


def _aggregate_multitask_curves(
    multitask_models: Sequence[nn.Module],
    multitask_data: Dict[str, object],
    multitask_groups: Dict[str, Dict[str, object]],
    feature_name: str,
    device: torch.device,
) -> Dict[str, object]:
  """Collect per-model and mean curves for one multitask feature."""
  group = multitask_groups[feature_name]
  x_positions = np.arange(len(group["categories"]), dtype=np.float32)
  women_curves = []
  men_curves = []
  for model in multitask_models:
    result = compute_categorical_curve(model, multitask_data["features"], group["indices"], device)
    women_curves.append(result["curve"][:, 0])
    men_curves.append(result["curve"][:, 1])
  women_array = np.stack(women_curves, axis=0)
  men_array = np.stack(men_curves, axis=0)
  return {
      "x": x_positions,
      "women_curves": women_array,
      "men_curves": men_array,
      "women_mean": np.mean(women_array, axis=0),
      "men_mean": np.mean(men_array, axis=0),
      "categories": group["categories"],
  }


def _compute_panel_limits(
    single_curves: Dict[str, Dict[str, object]],
    multitask_curves: Dict[str, Dict[str, object]],
) -> Dict[str, Sequence[float]]:
  """Use stable, feature-specific y-axis ranges for cleaner comparison."""
  limits = {}
  for feature_name, payload in single_curves.items():
    if payload["type"] == "numeric":
      values = np.concatenate([payload["curves"].reshape(-1), payload["mean"].reshape(-1)])
      bound = max(float(np.max(np.abs(values))) * 1.15, 0.25)
      if feature_name in {"age", "priors_count"}:
        bound = max(bound, 0.8)
      limits[feature_name] = (-bound, bound)
    else:
      values = np.concatenate([payload["curves"].reshape(-1), payload["mean"].reshape(-1)])
      bound = max(float(np.max(np.abs(values))) * 1.15, 0.18)
      limits[feature_name] = (-bound, bound)
  for feature_name, payload in multitask_curves.items():
    values = np.concatenate([
        payload["women_curves"].reshape(-1),
        payload["men_curves"].reshape(-1),
        payload["women_mean"].reshape(-1),
        payload["men_mean"].reshape(-1),
    ])
    bound = max(float(np.max(np.abs(values))) * 1.15, 0.18)
    limits[feature_name] = (-bound, bound)
  return limits


def inverse_numeric_axis(
    raw_values: Sequence[float],
    scaled_grid: np.ndarray,
) -> np.ndarray:
  """Invert min-max scaling back to the raw value range."""
  raw_array = np.asarray(raw_values, dtype=np.float32)
  min_value = float(np.min(raw_array))
  max_value = float(np.max(raw_array))
  return ((scaled_grid + 1.0) / 2.0) * (max_value - min_value) + min_value


def render_compas_figure(
    single_models: Sequence[nn.Module],
    multitask_models: Sequence[nn.Module],
    single_data: Dict[str, object],
    multitask_data: Dict[str, object],
    config: ExperimentConfig,
    device: torch.device,
) -> str:
  """Render a Figure 10 style summary plot for COMPAS."""
  single_groups = build_feature_groups(
      column_names=single_data["column_names"],
      raw_features=single_data["raw_features"],
      feature_columns=single_data["feature_columns"],
  )
  figure_path = osp.join(config.output_dir, "compas_figure10.png")
  fig = plt.figure(figsize=(16.2, 6.8))
  grid = fig.add_gridspec(2, 4, width_ratios=[1.05, 1.05, 1.05, 0.95], wspace=0.24, hspace=0.28)
  axes = np.array([[fig.add_subplot(grid[row, col]) for col in range(4)] for row in range(2)])

  single_feature_order = [name for name in SINGLE_TASK_PLOT_ORDER if name in single_groups]

  multitask_groups = build_feature_groups(
      column_names=multitask_data["column_names"],
      raw_features=multitask_data["raw_frame"][multitask_data["feature_columns"]].copy(),
      feature_columns=multitask_data["feature_columns"],
  )
  multitask_feature_order = [name for name in MULTITASK_PLOT_ORDER if name in multitask_groups]

  single_curves = {
      feature_name: _aggregate_single_task_curves(single_models, single_data, single_groups, feature_name, device)
      for feature_name in single_feature_order
  }
  multitask_curves = {
      feature_name: _aggregate_multitask_curves(multitask_models, multitask_data, multitask_groups, feature_name, device)
      for feature_name in multitask_feature_order
  }
  y_limits = _compute_panel_limits(single_curves, multitask_curves)

  single_panel_slots = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
  for feature_name, (row, col) in zip(single_feature_order, single_panel_slots):
    ax = axes[row, col]
    payload = single_curves[feature_name]
    group = single_groups[feature_name]
    raw_values = single_data["raw_features"][feature_name].to_numpy()
    _style_panel(
        ax=ax,
        title="",
        xlabel=FEATURE_LABELS[feature_name],
        ylim=y_limits[feature_name],
        show_ylabel=(col == 0),
    )
    _add_density_background(ax, group["type"], raw_values, group.get("categories"))
    if payload["type"] == "numeric":
      for curve in payload["curves"]:
        ax.plot(payload["x"], curve, color=BLUE, linewidth=1.0, alpha=0.08, zorder=2)
      ax.plot(payload["x"], payload["mean"], color=BLUE, linewidth=2.6, alpha=0.95, zorder=4)
    else:
      categories = _normalize_category_labels(payload["categories"])
      for curve in payload["curves"]:
        _plot_categorical_step(ax, payload["x"], curve, color=BLUE, linewidth=1.0, alpha=0.08)
      _plot_categorical_step(ax, payload["x"], payload["mean"], color=BLUE, linewidth=2.6, alpha=0.95)
      ax.set_xlim(-0.5, len(categories) - 0.5)
      ax.set_xticks(payload["x"])
      rotation = 0 if feature_name in {"sex", "c_charge_degree"} else 90
      ax.set_xticklabels(categories, rotation=rotation, ha="center")
    if payload["type"] == "numeric":
      ax.set_xlim(float(np.min(payload["x"])), float(np.max(payload["x"])))

  axes[1, 2].axis("off")

  for row, feature_name in enumerate(multitask_feature_order):
    ax = axes[row, 3]
    payload = multitask_curves[feature_name]
    group = multitask_groups[feature_name]
    raw_values = multitask_data["raw_frame"][feature_name].to_numpy()
    _style_panel(
        ax=ax,
        title="",
        xlabel=FEATURE_LABELS[feature_name],
        ylim=y_limits[feature_name],
        show_ylabel=False,
    )
    _add_density_background(ax, group["type"], raw_values, group.get("categories"))
    categories = _normalize_category_labels(payload["categories"])
    for curve in payload["women_curves"]:
      _plot_categorical_step(ax, payload["x"], curve, color=GREEN, linewidth=1.0, alpha=0.08)
    for curve in payload["men_curves"]:
      _plot_categorical_step(ax, payload["x"], curve, color=BLUE, linewidth=1.0, alpha=0.08)
    _plot_categorical_step(ax, payload["x"], payload["women_mean"], color=GREEN, linewidth=2.4, alpha=0.95, label="Women")
    _plot_categorical_step(ax, payload["x"], payload["men_mean"], color=BLUE, linewidth=2.4, alpha=0.95, label="Men")
    ax.set_xlim(-0.5, len(categories) - 0.5)
    ax.set_xticks(payload["x"])
    rotation = 0 if feature_name == "c_charge_degree" else 90
    ax.set_xticklabels(categories, rotation=rotation, ha="center")
    if row == 0:
      handles = [
          mlines.Line2D([], [], color=GREEN, linewidth=2.4, label="Women"),
          mlines.Line2D([], [], color=BLUE, linewidth=2.4, label="Men"),
      ]
      ax.legend(
          handles=handles,
          loc="upper right",
          frameon=True,
          fancybox=False,
          edgecolor="#333333",
          facecolor="white",
          fontsize=10,
          borderpad=0.4,
          handlelength=1.6,
          handletextpad=0.5,
      )

  axes[1, 3].axis("on")
  if "race" not in multitask_feature_order:
    axes[1, 3].axis("off")

  fig.text(0.43, 0.965, "Single Task NAM", ha="center", va="top", fontsize=19)
  fig.text(0.86, 0.965, "Multitask NAM", ha="center", va="top", fontsize=19)
  fig.add_artist(mlines.Line2D([0.735, 0.735], [0.1, 0.89], transform=fig.transFigure, color="#333333", linewidth=1.1))
  plt.tight_layout(rect=[0.02, 0.03, 0.98, 0.93])
  fig.savefig(figure_path, dpi=180)
  plt.close(fig)
  return figure_path


def train_models_for_figure(
    single_data: Dict[str, object],
    multitask_data: Dict[str, object],
    config: ExperimentConfig,
    device: torch.device,
) -> Dict[str, object]:
  """Train one fold worth of ensembles and render the figure."""
  labels = np.asarray(single_data["labels"], dtype=np.float32)
  task_index = np.asarray(single_data["task_index"], dtype=np.int64)
  strata = build_group_labels(labels, task_index)
  splitter = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=config.seed)
  fold_splits = list(splitter.split(single_data["features"], strata))
  if config.figure_fold < 1 or config.figure_fold > len(fold_splits):
    raise ValueError(f"figure_fold must be within [1, {len(fold_splits)}].")
  train_val_index, _ = fold_splits[config.figure_fold - 1]
  train_val_strata = strata[train_val_index]
  validation_splitter = StratifiedShuffleSplit(
      n_splits=1,
      test_size=0.125,
      random_state=config.seed + config.figure_fold,
  )
  train_rel_index, val_rel_index = next(validation_splitter.split(single_data["features"][train_val_index], train_val_strata))
  train_index = train_val_index[train_rel_index]
  val_index = train_val_index[val_rel_index]

  single_models = []
  multitask_models = []
  figure_dir = ensure_dir(osp.join(config.output_dir, "figure_training", f"fold_{config.figure_fold}"))
  for model_index in tqdm(range(config.n_models), desc=f"Figure fold {config.figure_fold} models"):
    single_model, _ = train_single_task_model(
        model_index=model_index,
        x_train=single_data["features"][train_index],
        y_train=labels[train_index],
        train_task_index=task_index[train_index],
        x_val=single_data["features"][val_index],
        y_val=labels[val_index],
        val_task_index=task_index[val_index],
        config=config,
        device=device,
        model_dir=ensure_dir(osp.join(figure_dir, "single_task", f"model_{model_index}")),
    )
    multitask_model, _ = train_multitask_model(
        model_index=model_index,
        x_train=multitask_data["features"][train_index],
        y_train=multitask_data["targets"][train_index],
        train_masks=multitask_data["masks"][train_index],
        train_task_index=task_index[train_index],
        x_val=multitask_data["features"][val_index],
        y_val=multitask_data["targets"][val_index],
        val_task_index=task_index[val_index],
        config=config,
        device=device,
        model_dir=ensure_dir(osp.join(figure_dir, "multitask", f"model_{model_index}")),
    )
    single_models.append(single_model)
    multitask_models.append(multitask_model)

  figure_path = render_compas_figure(
      single_models=single_models,
      multitask_models=multitask_models,
      single_data=single_data,
      multitask_data=multitask_data,
      config=config,
      device=device,
  )
  return {"figure_path": figure_path, "fold": config.figure_fold}


def save_json(payload: Dict[str, object], path: str) -> None:
  """Persist a JSON payload."""
  ensure_dir(osp.dirname(path))
  with open(path, "w", encoding="utf-8") as file_obj:
    json.dump(payload, file_obj, indent=2)


def save_summary_text(result: Dict[str, object], path: str) -> None:
  """Persist a human-readable summary."""
  lines = [
      "COMPAS Experiment Summary",
      f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
      "",
  ]
  if "summary" in result:
    summary = result["summary"]
    for model_name, display_name in (("single_task", "Single Task NAM"), ("multitask", "Multitask NAM")):
      lines.append(display_name)
      lines.append(
          "  Women: "
          f"{summary[model_name]['women_auc']['mean']:.3f} +/- {summary[model_name]['women_auc']['std']:.3f}"
      )
      lines.append(
          "  Men: "
          f"{summary[model_name]['men_auc']['mean']:.3f} +/- {summary[model_name]['men_auc']['std']:.3f}"
      )
      lines.append(
          "  Combined: "
          f"{summary[model_name]['combined_auc']['mean']:.3f} +/- {summary[model_name]['combined_auc']['std']:.3f}"
      )
      lines.append("")
  if "figure" in result:
    lines.append(f"Figure: {result['figure']['figure_path']}")
  with open(path, "w", encoding="utf-8") as file_obj:
    file_obj.write("\n".join(lines).rstrip() + "\n")


def main(argv: Sequence[str] | None = None) -> None:
  """CLI entrypoint."""
  args = build_parser().parse_args(argv)
  config = make_experiment_config(args)
  ensure_dir(config.output_dir)
  device = resolve_device(config.device)
  print('single-task exp begins')
  single_data = prepare_single_task_compas()
  print('multi-task exp begins')
  multitask_data = prepare_multitask_compas(config.include_sex_feature_for_multitask)

  result = {
      "config": {
          **config.__dict__,
          "resolved_device": str(device),
      },
      "generated_at": datetime.now().isoformat(timespec="seconds"),
  }

  if config.mode in {"all", "cv"}:
    result["cv"] = run_cross_validation(single_data, multitask_data, config, device)
    result["summary"] = result["cv"]["summary"]
  if config.mode in {"all", "figure"}:
    result["figure"] = train_models_for_figure(single_data, multitask_data, config, device)

  json_path = osp.join(config.output_dir, "compas_experiment_results.json")
  text_path = osp.join(config.output_dir, "compas_experiment_summary.txt")
  save_json(result, json_path)
  save_summary_text(result, text_path)
  print(f"Saved: {json_path}")
  print(f"Saved: {text_path}")
  if "figure" in result:
    print(f"Saved: {result['figure']['figure_path']}")


if __name__ == "__main__":
  main()
