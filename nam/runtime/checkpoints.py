"""Checkpoint helpers for PyTorch NAM models."""

from __future__ import annotations

import os
import os.path as osp
from typing import Any, Dict

import torch


def save_checkpoint(payload: Dict[str, Any], checkpoint_path: str) -> None:
  """Persist a checkpoint payload to disk."""
  os.makedirs(osp.dirname(checkpoint_path), exist_ok=True)
  torch.save(payload, checkpoint_path)


def load_checkpoint(checkpoint_path: str, map_location: torch.device | str = "cpu") -> Dict[str, Any]:
  """Load a checkpoint payload from disk."""
  return torch.load(checkpoint_path, map_location=map_location)


def find_checkpoint_path(model_dir: str) -> str:
  """Resolve the best available checkpoint path under a model directory."""
  best_checkpoint = osp.join(model_dir, "best_checkpoint", "model.pt")
  if osp.exists(best_checkpoint):
    return best_checkpoint
  fallback = osp.join(model_dir, "model.pt")
  if osp.exists(fallback):
    return fallback
  pt_files = sorted(
      filename
      for filename in [
          osp.join(model_dir, name)
          for name in os.listdir(model_dir)
      ]
      if filename.endswith(".pt")
  ) if osp.isdir(model_dir) else []
  if pt_files:
    return pt_files[-1]
  raise FileNotFoundError(f"No checkpoint found under: {model_dir}")
