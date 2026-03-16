"""Training loop implementations for NAM and DNN models."""

from __future__ import annotations

import os
import os.path as osp
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from neural_additive_models import models
from neural_additive_models.runtime.checkpoints import find_checkpoint_path
from neural_additive_models.runtime.checkpoints import load_checkpoint
from neural_additive_models.runtime.checkpoints import save_checkpoint
from neural_additive_models.runtime.device import resolve_device
from neural_additive_models.training.data import create_eval_loader
from neural_additive_models.training.data import create_train_loader
from neural_additive_models.training.losses import calculate_loss
from neural_additive_models.training.metrics import calculate_metric


@dataclass
class TrainingConfig:
  """Training configuration used by CLI and tests."""

  training_epochs: int
  learning_rate: float = 1e-2
  output_regularization: float = 0.0
  l2_regularization: float = 0.0
  batch_size: int = 1024
  decay_rate: float = 0.995
  dropout: float = 0.5
  feature_dropout: float = 0.0
  num_basis_functions: int = 1000
  units_multiplier: int = 2
  n_models: int = 1
  activation: str = "exu"
  regression: bool = False
  debug: bool = False
  shallow: bool = False
  use_dnn: bool = False
  early_stopping_epochs: int = 60
  save_checkpoint_every_n_epochs: int = 10
  max_checkpoints_to_keep: int = 1
  device: str = "auto"
  tf_seed: int = 1


def infer_num_units(
    x_train: np.ndarray,
    num_basis_functions: int = 64,
    units_multiplier: int = 2,
) -> List[int]:
  """Infer feature net widths using the legacy project heuristic."""
  num_unique_values = [len(np.unique(x_train[:, index])) for index in range(x_train.shape[1])]
  return [
      min(num_basis_functions, max(unique_value_count * units_multiplier, 1))
      for unique_value_count in num_unique_values
  ]


def create_nam_model(
    x_train: np.ndarray,
    dropout: float,
    feature_dropout: float = 0.0,
    num_basis_functions: int = 1000,
    units_multiplier: int = 2,
    activation: str = "exu",
    shallow: bool = True,
) -> models.NAM:
  """Create a NAM model using dataset-derived per-feature widths."""
  num_units = infer_num_units(
      x_train=x_train,
      num_basis_functions=num_basis_functions,
      units_multiplier=units_multiplier,
  )
  return models.NAM(
      num_inputs=x_train.shape[1],
      num_units=num_units,
      dropout=dropout,
      feature_dropout=feature_dropout,
      activation=activation,
      shallow=shallow,
  )


def create_model(config: TrainingConfig, x_train: np.ndarray) -> nn.Module:
  """Build the requested architecture."""
  if config.use_dnn:
    return models.DNN(input_dim=x_train.shape[1], dropout=config.dropout)
  return create_nam_model(
      x_train=x_train,
      dropout=config.dropout,
      feature_dropout=config.feature_dropout,
      num_basis_functions=config.num_basis_functions,
      units_multiplier=config.units_multiplier,
      activation=config.activation,
      shallow=config.shallow,
  )


def _predict(model: nn.Module, features: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
  """Run batched prediction."""
  model.eval()
  loader = create_eval_loader(features, batch_size=batch_size)
  outputs = []
  with torch.no_grad():
    for batch_features, _ in loader:
      batch_features = batch_features.to(device)
      outputs.append(model(batch_features, training=False).detach().cpu().numpy())
  if not outputs:
    return np.empty((0,), dtype=np.float32)
  return np.concatenate(outputs, axis=0)


def evaluate_model(
    model: nn.Module,
    features: np.ndarray,
    targets: np.ndarray,
    regression: bool,
    batch_size: int,
    device: torch.device,
) -> float:
  """Evaluate a model on a numpy dataset."""
  predictions = _predict(model, features, batch_size=batch_size, device=device)
  return calculate_metric(targets, predictions, regression=regression)


def _checkpoint_payload(
    model: nn.Module,
    config: TrainingConfig,
    epoch: int,
    metric_value: float,
) -> Dict[str, object]:
  """Build a serializable checkpoint payload."""
  return {
      "epoch": epoch,
      "metric_value": metric_value,
      "state_dict": model.state_dict(),
      "config": dict(config.__dict__),
      "model_type": model.__class__.__name__,
  }


def _train_single_model(
    model_index: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    logdir: str,
    config: TrainingConfig,
    device: torch.device,
) -> Tuple[float, float, int]:
  """Train one model and return the best train metric, validation metric, and epoch."""
  model = create_model(config, x_train).to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
  scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.decay_rate)
  train_loader = create_train_loader(
      x_train=x_train,
      y_train=y_train,
      batch_size=min(config.batch_size, x_train.shape[0]),
      regression=config.regression,
  )
  metric_name = "RMSE" if config.regression else "AUROC"
  compare = (lambda current, best: current < best) if config.regression else (lambda current, best: current > best)
  best_validation_metric = np.inf if config.regression else 0.0
  best_train_metric = np.inf if config.regression else 0.0
  best_epoch = 0
  model_dir = osp.join(logdir, f"model_{model_index}")
  best_dir = osp.join(model_dir, "best_checkpoint")
  os.makedirs(best_dir, exist_ok=True)
  saved_checkpoints: List[str] = []

  for epoch in tqdm(range(1, config.training_epochs + 1), desc=f"Model {model_index} Training"):
    model.train()
    for batch_features, batch_targets in train_loader:
      batch_features = batch_features.to(device)
      batch_targets = batch_targets.to(device)
      optimizer.zero_grad()
      loss = calculate_loss(
          model=model,
          inputs=batch_features,
          targets=batch_targets,
          regression=config.regression,
          output_regularization=config.output_regularization,
          l2_regularization=config.l2_regularization,
          use_dnn=config.use_dnn,
      )
      loss.backward()
      optimizer.step()
    scheduler.step()

    if epoch % config.save_checkpoint_every_n_epochs != 0:
      continue

    validation_metric = evaluate_model(
        model=model,
        features=x_validation,
        targets=y_validation,
        regression=config.regression,
        batch_size=config.batch_size,
        device=device,
    )
    if config.debug:
      print(f"Model {model_index} epoch {epoch} {metric_name} val {validation_metric:.4f}")
    checkpoint_path = osp.join(model_dir, f"checkpoint_epoch_{epoch}.pt")
    save_checkpoint(_checkpoint_payload(model, config, epoch, validation_metric), checkpoint_path)
    saved_checkpoints.append(checkpoint_path)
    while len(saved_checkpoints) > config.max_checkpoints_to_keep:
      old_checkpoint = saved_checkpoints.pop(0)
      if osp.exists(old_checkpoint):
        os.remove(old_checkpoint)
    if compare(validation_metric, best_validation_metric):
      best_validation_metric = validation_metric
      best_train_metric = evaluate_model(
          model=model,
          features=x_train,
          targets=y_train,
          regression=config.regression,
          batch_size=config.batch_size,
          device=device,
      )
      best_epoch = epoch
      save_checkpoint(_checkpoint_payload(model, config, epoch, validation_metric), osp.join(best_dir, "model.pt"))
    elif best_epoch and best_epoch + config.early_stopping_epochs < epoch:
      break

  if best_epoch == 0:
    best_train_metric = evaluate_model(
        model=model,
        features=x_train,
        targets=y_train,
        regression=config.regression,
        batch_size=config.batch_size,
        device=device,
    )
    best_validation_metric = evaluate_model(
        model=model,
        features=x_validation,
        targets=y_validation,
        regression=config.regression,
        batch_size=config.batch_size,
        device=device,
    )
    best_epoch = config.training_epochs
    save_checkpoint(
        _checkpoint_payload(model, config, best_epoch, best_validation_metric),
        osp.join(best_dir, "model.pt"),
    )
  return float(best_train_metric), float(best_validation_metric), int(best_epoch)


def _load_best_model(
    model_dir: str,
    config: TrainingConfig,
    x_train: np.ndarray,
    device: torch.device,
) -> nn.Module:
  """Restore the best checkpoint for a trained model."""
  checkpoint_path = find_checkpoint_path(model_dir)
  checkpoint = load_checkpoint(checkpoint_path, map_location=device)
  model = create_model(config, x_train).to(device)
  model.load_state_dict(checkpoint["state_dict"])
  model.eval()
  return model


def train_ensemble(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    logdir: str,
    config: TrainingConfig,
    x_test: np.ndarray | None = None,
    y_test: np.ndarray | None = None,
    return_test_metric: bool = False,
) -> Tuple[float, float] | Tuple[float, float, float]:
  """Train an ensemble and return mean train and validation metrics."""
  os.makedirs(logdir, exist_ok=True)
  device = resolve_device(config.device)
  best_train_metrics = []
  best_validation_metrics = []
  best_epochs = []
  for model_index in range(config.n_models):
    model_train_metric, model_validation_metric, best_epoch = _train_single_model(
        model_index=model_index,
        x_train=x_train,
        y_train=y_train,
        x_validation=x_validation,
        y_validation=y_validation,
        logdir=logdir,
        config=config,
        device=device,
    )
    best_train_metrics.append(model_train_metric)
    best_validation_metrics.append(model_validation_metric)
    best_epochs.append(best_epoch)
  for index, (epoch, train_metric, validation_metric) in enumerate(
      zip(best_epochs, best_train_metrics, best_validation_metrics)
  ):
    metric_name = "RMSE" if config.regression else "AUROC"
    print(
        f"Model {index}: Best Epoch {epoch}, Individual {metric_name}: "
        f"Train {train_metric:.4f}, Validation {validation_metric:.4f}"
    )
  mean_train_metric = float(np.mean(best_train_metrics))
  mean_validation_metric = float(np.mean(best_validation_metrics))
  if not return_test_metric:
    return mean_train_metric, mean_validation_metric
  if x_test is None or y_test is None:
    raise ValueError("x_test and y_test must be provided when return_test_metric=True")
  test_scores = []
  for model_index in range(config.n_models):
    model = _load_best_model(
        model_dir=osp.join(logdir, f"model_{model_index}"),
        config=config,
        x_train=x_train,
        device=device,
    )
    test_scores.append(
        evaluate_model(
            model=model,
            features=x_test,
            targets=y_test,
            regression=config.regression,
            batch_size=config.batch_size,
            device=device,
        )
    )
  return mean_train_metric, mean_validation_metric, float(np.mean(test_scores))
