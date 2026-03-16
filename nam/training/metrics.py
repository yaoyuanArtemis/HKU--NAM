"""Metric helpers for training and evaluation."""

from __future__ import annotations

import numpy as np
from sklearn import metrics as sk_metrics
import torch


def sigmoid(x):
  """Apply a numerically stable sigmoid to numpy-like values."""
  x = np.asarray(x, dtype=np.float64)
  result = np.empty_like(x, dtype=np.float64)
  positive = x >= 0
  result[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
  exp_x = np.exp(x[~positive])
  result[~positive] = exp_x / (1.0 + exp_x)
  return result


def rmse(y_true, y_pred) -> float:
  """Return root mean squared error."""
  return float(np.sqrt(sk_metrics.mean_squared_error(y_true, y_pred)))


def calculate_metric(y_true, predictions, regression: bool = True) -> float:
  """Compute the project evaluation metric."""
  if regression:
    return rmse(y_true, predictions)
  return float(sk_metrics.roc_auc_score(y_true, sigmoid(predictions)))


def detach_to_numpy(value: torch.Tensor) -> np.ndarray:
  """Detach a tensor and convert it to numpy."""
  return value.detach().cpu().numpy()
