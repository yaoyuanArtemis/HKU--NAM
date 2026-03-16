"""Helpers for device selection."""

from __future__ import annotations

import torch


def resolve_device(device: str = "auto") -> torch.device:
  """Resolve the runtime device according to project policy."""
  normalized = device.lower()
  if normalized not in {"auto", "cuda", "mps", "cpu"}:
    raise ValueError(f"Unsupported device value: {device}")
  if normalized == "cuda":
    if not torch.cuda.is_available():
      raise RuntimeError("CUDA was requested but is not available.")
    return torch.device("cuda")
  if normalized == "mps":
    if not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available():
      raise RuntimeError("MPS was requested but is not available.")
    return torch.device("mps")
  if normalized == "cpu":
    return torch.device("cpu")
  if torch.cuda.is_available():
    return torch.device("cuda")
  if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    return torch.device("mps")
  return torch.device("cpu")
