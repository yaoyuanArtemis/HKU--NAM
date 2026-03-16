#!/usr/bin/env python
# coding=utf-8
"""Dataset-generic NAM vs NAM+FM experiment."""

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

import numpy as np
import torch
from torch import nn
from neural_additive_models import data_utils
from neural_additive_models import models
from neural_additive_models.experiments.nam.train import str2bool
from neural_additive_models.runtime import resolve_device
from neural_additive_models.runtime import save_checkpoint
from neural_additive_models.training.data import create_eval_loader
from neural_additive_models.training.data import create_train_loader
from neural_additive_models.training.metrics import calculate_metric
from neural_additive_models.training.trainer import infer_num_units
from tqdm.auto import tqdm


REGRESSION_DATASETS = {"Fico", "Housing"}


@dataclass
class ExperimentConfig:
  """Hyperparameters for the NAM vs NAM+FM experiment."""

  dataset_name: str
  regression: bool
  training_epochs: int
  learning_rate: float
  batch_size: int
  decay_rate: float
  dropout: float
  feature_dropout: float
  num_basis_functions: int
  units_multiplier: int
  fm_rank: int
  activation: str
  shallow: bool
  output_regularization: float
  l2_regularization: float
  early_stopping_epochs: int
  n_models: int
  n_folds: int
  seed: int
  validation_size: float
  device: str
  output_dir: str


def build_parser() -> argparse.ArgumentParser:
  """Create the CLI parser."""
  parser = argparse.ArgumentParser(description="Run a dataset-generic NAM vs NAM+FM experiment.")
  parser.add_argument("--dataset_name", type=str, default="Recidivism")#Recidivism Housing Fico
  parser.add_argument("--regression", type=str2bool, nargs="?", const=False, default=None)
  parser.add_argument("--training_epochs", type=int, default=1000)
  parser.add_argument("--learning_rate", type=float, default=1e-2)
  parser.add_argument("--batch_size", type=int, default=1024)
  parser.add_argument("--decay_rate", type=float, default=0.995)
  parser.add_argument("--dropout", type=float, default=0.15)
  parser.add_argument("--feature_dropout", type=float, default=0.0)
  parser.add_argument("--num_basis_functions", type=int, default=64)
  parser.add_argument("--units_multiplier", type=int, default=2)
  parser.add_argument("--fm_rank", type=int, default=12)
  parser.add_argument("--activation", type=str, default="exu")
  parser.add_argument("--shallow", type=str2bool, nargs="?", const=True, default=True)
  parser.add_argument("--output_regularization", type=float, default=0.0)
  parser.add_argument("--l2_regularization", type=float, default=0.0)
  parser.add_argument("--early_stopping_epochs", type=int, default=20)
  parser.add_argument("--n_models", type=int, default=10)
  parser.add_argument("--n_folds", type=int, default=5)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--validation_size", type=float, default=0.125)
  parser.add_argument("--device", type=str, default="cpu", choices=["auto", "cuda", "mps", "cpu"])
  parser.add_argument("--output_dir", type=str, default=None)
  return parser


def is_regression_dataset(dataset_name: str) -> bool:
  """Infer the task type from the dataset name."""
  return dataset_name in REGRESSION_DATASETS


def make_experiment_config(args: argparse.Namespace) -> ExperimentConfig:
  """Convert CLI args into a typed config."""
  regression = is_regression_dataset(args.dataset_name) if args.regression is None else args.regression
  output_dir = args.output_dir
  if output_dir is None:
    output_dir = osp.join(_repo_root(), "outputs", "nam_vs_fm", args.dataset_name)
  return ExperimentConfig(
      dataset_name=args.dataset_name,
      regression=regression,
      training_epochs=args.training_epochs,
      learning_rate=args.learning_rate,
      batch_size=args.batch_size,
      decay_rate=args.decay_rate,
      dropout=args.dropout,
      feature_dropout=args.feature_dropout,
      num_basis_functions=args.num_basis_functions,
      units_multiplier=args.units_multiplier,
      fm_rank=args.fm_rank,
      activation=args.activation,
      shallow=args.shallow,
      output_regularization=args.output_regularization,
      l2_regularization=args.l2_regularization,
      early_stopping_epochs=args.early_stopping_epochs,
      n_models=args.n_models,
      n_folds=args.n_folds,
      seed=args.seed,
      validation_size=args.validation_size,
      device=args.device,
      output_dir=output_dir,
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


def prepare_dataset(dataset_name: str) -> Dict[str, object]:
  """Load a dataset using the shared project preprocessing."""
  features, labels, column_names = data_utils.load_dataset(dataset_name)
  return {
      "features": np.asarray(features, dtype=np.float32),
      "labels": np.asarray(labels, dtype=np.float32),
      "column_names": column_names,
  }


def feature_output_regularization(model: nn.Module, inputs: torch.Tensor) -> torch.Tensor:
  """Penalize the mean squared additive NAM contributions."""
  per_feature_outputs = model.calc_outputs(inputs, training=False)
  penalties = [torch.mean(outputs.square()) for outputs in per_feature_outputs]
  return torch.stack(penalties).mean()


def weight_decay(model: nn.Module, num_networks: int) -> torch.Tensor:
  """Compute L2 regularization across trainable parameters."""
  penalties = [0.5 * parameter.square().sum() for parameter in model.parameters() if parameter.requires_grad]
  if not penalties:
    return torch.tensor(0.0)
  return torch.stack(penalties).sum() / max(num_networks, 1)


def build_nam_model(x_train: np.ndarray, config: ExperimentConfig) -> models.NAM:
  """Create the NAM baseline."""
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


def build_factorized_nam_model(x_train: np.ndarray, config: ExperimentConfig) -> models.FactorizedNAM:
  """Create the NAM + FM interaction model."""
  num_units = infer_num_units(
      x_train=x_train,
      num_basis_functions=config.num_basis_functions,
      units_multiplier=config.units_multiplier,
  )
  return models.FactorizedNAM(
      num_inputs=x_train.shape[1],
      num_units=num_units,
      fm_rank=config.fm_rank,
      shallow=config.shallow,
      feature_dropout=config.feature_dropout,
      dropout=config.dropout,
      activation=config.activation,
  )


def predict_values(
    model: nn.Module,
    features: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
  """Predict logits or regression outputs in batches."""
  loader = create_eval_loader(features, batch_size=batch_size)
  outputs = []
  model.eval()
  with torch.no_grad():
    for batch_features, _ in loader:
      outputs.append(model(batch_features.to(device), training=False).detach().cpu().numpy())
  return np.concatenate(outputs, axis=0)


def _checkpoint_payload(model: nn.Module, epoch: int, metric_value: float) -> Dict[str, object]:
  """Build a minimal checkpoint payload."""
  return {
      "epoch": epoch,
      "metric_value": metric_value,
      "state_dict": model.state_dict(),
      "model_type": model.__class__.__name__,
  }


def calculate_training_loss(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    config: ExperimentConfig,
) -> torch.Tensor:
  """Compute the task loss and optional regularization terms."""
  predictions = model(inputs, training=True)
  if config.regression:
    loss = nn.functional.mse_loss(predictions, targets)
  else:
    loss = nn.functional.binary_cross_entropy_with_logits(predictions, targets)
  if config.output_regularization > 0:
    loss = loss + config.output_regularization * feature_output_regularization(model, inputs)
  if config.l2_regularization > 0:
    loss = loss + config.l2_regularization * weight_decay(model, num_networks=len(model.feature_nns))
  return loss


def is_better_metric(current: float, best: float, regression: bool) -> bool:
  """Compare metrics using the task-specific optimization direction."""
  return current < best if regression else current > best


def metric_name(regression: bool) -> str:
  """Return the display name of the primary metric."""
  return "RMSE" if regression else "AUROC"


def train_model(
    model_kind: str,
    model_index: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    config: ExperimentConfig,
    device: torch.device,
    model_dir: str,
):
  """Train one model and return the best checkpointed model."""
  set_torch_seed(config.seed + model_index)
  if model_kind == "nam":
    model = build_nam_model(x_train, config).to(device)
  elif model_kind == "factorized_nam":
    model = build_factorized_nam_model(x_train, config).to(device)
  else:
    raise ValueError(f"Unsupported model kind: {model_kind}")

  optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
  scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.decay_rate)
  train_loader = create_train_loader(
      x_train=x_train,
      y_train=y_train,
      batch_size=min(config.batch_size, x_train.shape[0]),
      regression=config.regression,
  )

  best_metric = np.inf if config.regression else -np.inf
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
      loss = calculate_training_loss(model, batch_features, batch_targets, config)
      loss.backward()
      optimizer.step()
    scheduler.step()

    val_predictions = predict_values(model, x_val, config.batch_size, device)
    validation_metric = float(calculate_metric(y_val, val_predictions, regression=config.regression))
    if is_better_metric(validation_metric, best_metric, config.regression):
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
    fallback_metric = float(calculate_metric(y_val, predict_values(model, x_val, config.batch_size, device), regression=config.regression))
    save_checkpoint(_checkpoint_payload(model, config.training_epochs, fallback_metric), osp.join(best_dir, "model.pt"))
    best_metric = fallback_metric
  model.load_state_dict(best_state)
  model.eval()
  return model, {"best_epoch": best_epoch or config.training_epochs, "best_validation_metric": float(best_metric)}


def ensemble_mean(predictions: Sequence[np.ndarray]) -> np.ndarray:
  """Average a list of per-model prediction arrays."""
  return np.mean(np.stack(predictions, axis=0), axis=0)


def summarize_metric_list(values: Iterable[float]) -> Dict[str, float]:
  """Return mean and std summary for a list of metrics."""
  array = np.asarray(list(values), dtype=np.float64)
  return {"mean": float(np.mean(array)), "std": float(np.std(array, ddof=1) if len(array) > 1 else 0.0)}


def run_cross_validation(
    dataset: Dict[str, object],
    config: ExperimentConfig,
    device: torch.device,
) -> Dict[str, object]:
  """Run cross-validation using the shared project splitting rules."""
  features = np.asarray(dataset["features"], dtype=np.float32)
  labels = np.asarray(dataset["labels"], dtype=np.float32)
  cv_results = {"nam": [], "factorized_nam": []}

  print(f"Dataset: {config.dataset_name}, Size: {features.shape[0]}")
  for fold_index in range(1, config.n_folds + 1):
    print(f"Cross-val fold: {fold_index}/{config.n_folds}")
    fold_dir = ensure_dir(osp.join(config.output_dir, "training", f"fold_{fold_index}"))
    (x_train_all, y_train_all), (x_test, y_test) = data_utils.get_train_test_fold(
        features,
        labels,
        fold_num=fold_index,
        num_folds=config.n_folds,
        stratified=not config.regression,
        random_state=config.seed,
    )
    validation_gen = data_utils.split_training_dataset(
        x_train_all,
        y_train_all,
        n_splits=1,
        stratified=not config.regression,
        test_size=config.validation_size,
        random_state=config.seed + fold_index,
    )
    (x_train, y_train), (x_val, y_val) = next(validation_gen)

    for model_kind in ("nam", "factorized_nam"):
      model_predictions = []
      model_metadata = []
      model_fold_dir = ensure_dir(osp.join(fold_dir, model_kind))
      for model_index in tqdm(range(config.n_models), desc=f"Fold {fold_index} {model_kind}"):
        model, model_info = train_model(
            model_kind=model_kind,
            model_index=model_index,
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            config=config,
            device=device,
            model_dir=ensure_dir(osp.join(model_fold_dir, f"model_{model_index}")),
        )
        predictions = predict_values(model, x_test, config.batch_size, device)
        model_predictions.append(predictions)
        model_metadata.append(model_info)

      ensemble_predictions = ensemble_mean(model_predictions)
      fold_metric = float(calculate_metric(y_test, ensemble_predictions, regression=config.regression))
      cv_results[model_kind].append({
          "fold": fold_index,
          "test_metric": fold_metric,
          "models": model_metadata,
      })

  summary = {
      "metric_name": metric_name(config.regression),
      "optimize_mode": "min" if config.regression else "max",
      "nam": {
          "metric": summarize_metric_list([fold["test_metric"] for fold in cv_results["nam"]]),
      },
      "factorized_nam": {
          "metric": summarize_metric_list([fold["test_metric"] for fold in cv_results["factorized_nam"]]),
      },
  }
  summary["delta_metric"] = (
      summary["factorized_nam"]["metric"]["mean"] -
      summary["nam"]["metric"]["mean"]
  )
  return {"folds": cv_results, "summary": summary}


def save_json(payload: Dict[str, object], path: str) -> None:
  """Persist a JSON payload."""
  ensure_dir(osp.dirname(path))
  with open(path, "w", encoding="utf-8") as file_obj:
    json.dump(payload, file_obj, indent=2)


def save_summary_text(result: Dict[str, object], path: str) -> None:
  """Persist a human-readable summary."""
  summary = result["summary"]
  metric_label = summary["metric_name"]
  direction = "lower is better" if summary["optimize_mode"] == "min" else "higher is better"
  lines = [
      "NAM vs NAM+FM Experiment Summary",
      f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
      f"Dataset: {result['config']['dataset_name']}",
      f"Primary metric: {metric_label} ({direction})",
      "",
      "NAM",
      "  Metric: "
      f"{summary['nam']['metric']['mean']:.3f} +/- {summary['nam']['metric']['std']:.3f}",
      "",
      "NAM + FactorizedMachine",
      "  Metric: "
      f"{summary['factorized_nam']['metric']['mean']:.3f} +/- "
      f"{summary['factorized_nam']['metric']['std']:.3f}",
      "",
      f"Delta {metric_label} (NAM+FM - NAM): {summary['delta_metric']:.4f}",
  ]
  with open(path, "w", encoding="utf-8") as file_obj:
    file_obj.write("\n".join(lines).rstrip() + "\n")


def main(argv: Sequence[str] | None = None) -> None:
  """CLI entrypoint."""
  args = build_parser().parse_args(argv)
  config = make_experiment_config(args)
  ensure_dir(config.output_dir)
  device = resolve_device(config.device)
  dataset = prepare_dataset(config.dataset_name)
  cv_result = run_cross_validation(dataset, config, device)
  result = {
      "experiment_name": "nam_vs_factorized_nam",
      "config": {
          **config.__dict__,
          "resolved_device": str(device),
      },
      "generated_at": datetime.now().isoformat(timespec="seconds"),
      **cv_result,
  }
  json_path = osp.join(config.output_dir, f"{config.dataset_name}_FM_results.json")
  text_path = osp.join(config.output_dir, f"{config.dataset_name}_FM_summary.txt")
  save_json(result, json_path)
  save_summary_text(result, text_path)
  print(f"Saved: {json_path}")
  print(f"Saved: {text_path}")


if __name__ == "__main__":
  main()
