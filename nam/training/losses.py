"""Loss and regularization helpers."""

from __future__ import annotations

import torch
from torch import Tensor
from torch import nn

from neural_additive_models.models import NAM


def feature_output_regularization(model: NAM, inputs: Tensor) -> Tensor:
  """Penalize the mean squared contribution of each feature net."""
  per_feature_outputs = model.calc_outputs(inputs, training=False)
  penalties = [torch.mean(outputs.square()) for outputs in per_feature_outputs]
  return torch.stack(penalties).mean()


def weight_decay(model: nn.Module, num_networks: int = 1) -> Tensor:
  """Penalize the L2 norm of trainable parameters."""
  l2_losses = [0.5 * parameter.square().sum() for parameter in model.parameters() if parameter.requires_grad]
  if not l2_losses:
    return torch.tensor(0.0)
  return torch.stack(l2_losses).sum() / max(num_networks, 1)


def calculate_loss(
    model: nn.Module,
    inputs: Tensor,
    targets: Tensor,
    regression: bool,
    output_regularization: float = 0.0,
    l2_regularization: float = 0.0,
    use_dnn: bool = False,
) -> Tensor:
  """Compute the primary loss with optional penalties."""
  predictions = model(inputs, training=True)
  if regression:
    base_loss = nn.functional.mse_loss(predictions, targets)
  else:
    base_loss = nn.functional.binary_cross_entropy_with_logits(predictions, targets)
  regularization = predictions.new_tensor(0.0)
  if output_regularization > 0 and not use_dnn and isinstance(model, NAM):
    regularization = regularization + output_regularization * feature_output_regularization(model, inputs)
  if l2_regularization > 0:
    num_networks = 1 if use_dnn else len(getattr(model, "feature_nns", [None]))
    regularization = regularization + l2_regularization * weight_decay(model, num_networks=num_networks)
  return base_loss + regularization
