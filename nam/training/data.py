"""PyTorch dataset and dataloader helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import WeightedRandomSampler


class TabularDataset(Dataset):
  """Dataset wrapper for dense tabular arrays."""

  def __init__(self, features: np.ndarray, targets: np.ndarray) -> None:
    self.features = torch.as_tensor(features, dtype=torch.float32)
    self.targets = torch.as_tensor(targets, dtype=torch.float32)

  def __len__(self) -> int:
    return int(self.features.shape[0])

  def __getitem__(self, index: int):
    return self.features[index], self.targets[index]


def create_train_loader(
    x_train: np.ndarray,
    y_train: np.ndarray,
    batch_size: int,
    regression: bool,
) -> DataLoader:
  """Create the train dataloader with balanced sampling for classification."""
  dataset = TabularDataset(x_train, y_train)
  if regression:
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
  labels = y_train.astype(np.int64)
  class_counts = np.bincount(labels, minlength=2)
  class_weights = np.zeros_like(class_counts, dtype=np.float64)
  non_zero = class_counts > 0
  class_weights[non_zero] = 1.0 / class_counts[non_zero]
  sample_weights = class_weights[labels]
  sampler = WeightedRandomSampler(
      weights=torch.as_tensor(sample_weights, dtype=torch.double),
      num_samples=len(sample_weights),
      replacement=True,
  )
  return DataLoader(dataset, batch_size=batch_size, sampler=sampler)


def create_eval_loader(
    features: np.ndarray,
    targets: np.ndarray | None = None,
    batch_size: int = 1024,
) -> DataLoader:
  """Create a deterministic evaluation dataloader."""
  if targets is None:
    targets = np.zeros(features.shape[0], dtype=np.float32)
  dataset = TabularDataset(features, targets)
  return DataLoader(dataset, batch_size=batch_size, shuffle=False)
