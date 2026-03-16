"""PyTorch implementations of NAM building blocks."""

from __future__ import annotations

from typing import List, Sequence

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F


def exu(x: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
  """Apply the ExU transformation."""
  return torch.exp(weight) * (x - bias)


def relu(x: Tensor, weight: Tensor, bias: Tensor) -> Tensor:
  """Apply the ReLU transformation."""
  return F.relu(weight * (x - bias))


def relu_n(x: Tensor, n: float = 1.0) -> Tensor:
  """Apply ReLU clipped at ``n``."""
  return torch.clamp(x, min=0.0, max=n)


class ActivationLayer(nn.Module):
  """Activation layer used by the first hidden layer of each feature net."""

  def __init__(
      self,
      num_units: int,
      activation: str = "exu",
      in_features: int = 1,
  ) -> None:
    super().__init__()
    self.num_units = num_units
    self.activation = activation
    self.in_features = in_features
    self.beta = nn.Parameter(torch.empty(in_features, num_units))
    self.c = nn.Parameter(torch.empty(1, num_units))
    self.reset_parameters()

  def reset_parameters(self) -> None:
    """Initialize learnable parameters."""
    if self.activation == "relu":
      nn.init.xavier_uniform_(self.beta)
    elif self.activation == "exu":
      nn.init.trunc_normal_(self.beta, mean=4.0, std=0.5, a=3.0, b=5.0)
    else:
      raise ValueError(f"{self.activation} is not a valid activation")
    nn.init.trunc_normal_(self.c, mean=0.0, std=0.5, a=-1.0, b=1.0)

  def forward(self, x: Tensor) -> Tensor:
    """Return hidden activations for the input feature values."""
    if x.ndim == 1:
      x = x.unsqueeze(1)
    if x.shape[1] != self.in_features:
      raise ValueError(
          f"Expected input with {self.in_features} feature columns, got {x.shape[1]}."
      )
    if self.in_features != 1:
      raise ValueError("ActivationLayer currently supports one feature per feature network.")
    expanded_x = x.expand(-1, self.num_units)
    bias = self.c.expand(x.shape[0], -1)
    weight = self.beta.expand_as(expanded_x)
    if self.activation == "relu":
      return relu(expanded_x, weight, bias)
    return relu_n(exu(expanded_x, weight, bias))


class FeatureNN(nn.Module):
  """Per-feature neural network used by NAM."""

  def __init__(
      self,
      num_units: int,
      dropout: float = 0.5,
      shallow: bool = True,
      feature_num: int = 0,
      activation: str = "exu",
      output_dim: int = 1,
  ) -> None:
    super().__init__()
    self.dropout = dropout
    self.shallow = shallow
    self.feature_num = feature_num
    self.activation = activation
    self.output_dim = output_dim
    self.activation_layer = ActivationLayer(num_units=num_units, activation=activation)
    self.hidden_layers = nn.ModuleList([self.activation_layer])
    if not shallow:
      self.hidden_layers.append(nn.Linear(num_units, 64))
      self.hidden_layers.append(nn.Linear(64, 32))
      self.output_layer = nn.Linear(32, output_dim, bias=False)
    else:
      self.output_layer = nn.Linear(num_units, output_dim, bias=False)

  def forward(self, x: Tensor, training: bool | None = None) -> Tensor:
    """Return the scalar contribution for the corresponding feature."""
    should_train = self.training if training is None else training
    hidden = self.activation_layer(x)
    hidden = F.dropout(hidden, p=self.dropout, training=should_train)
    if not self.shallow:
      hidden = F.dropout(F.relu(self.hidden_layers[1](hidden)), p=self.dropout, training=should_train)
      hidden = F.dropout(F.relu(self.hidden_layers[2](hidden)), p=self.dropout, training=should_train)
    outputs = self.output_layer(hidden)
    if self.output_dim == 1:
      return outputs.squeeze(-1)
    return outputs


class NAM(nn.Module):
  """Neural additive model implemented with PyTorch."""

  def __init__(
      self,
      num_inputs: int,
      num_units: int | Sequence[int],
      shallow: bool = True,
      feature_dropout: float = 0.0,
      dropout: float = 0.0,
      activation: str = "exu",
  ) -> None:
    super().__init__()
    self.num_inputs = num_inputs
    if isinstance(num_units, int):
      self.num_units = [num_units for _ in range(num_inputs)]
    else:
      self.num_units = list(num_units)
    if len(self.num_units) != num_inputs:
      raise ValueError("The number of unit values must match the number of inputs.")
    self.shallow = shallow
    self.feature_dropout = feature_dropout
    self.dropout = dropout
    self.activation = activation
    self.feature_nns = nn.ModuleList([
        FeatureNN(
            num_units=self.num_units[index],
            dropout=dropout,
            shallow=shallow,
            feature_num=index,
            activation=activation,
            output_dim=1,
        )
        for index in range(num_inputs)
    ])
    self.bias = nn.Parameter(torch.zeros(1))

  def calc_outputs(self, x: Tensor, training: bool | None = None) -> List[Tensor]:
    """Return one output tensor per input feature."""
    if x.ndim != 2:
      raise ValueError(f"Expected rank-2 input, got shape {tuple(x.shape)}")
    return [
        feature_nn(feature_values, training=training)
        for feature_nn, feature_values in zip(self.feature_nns, torch.split(x, 1, dim=1))
    ]

  def forward(self, x: Tensor, training: bool | None = None) -> Tensor:
    """Return the model prediction."""
    should_train = self.training if training is None else training
    individual_outputs = self.calc_outputs(x, training=should_train)
    stacked_outputs = torch.stack(individual_outputs, dim=-1)
    dropped_outputs = F.dropout(
        stacked_outputs,
        p=self.feature_dropout,
        training=should_train,
    )
    return dropped_outputs.sum(dim=-1) + self.bias


class MultiTaskNAM(nn.Module):
  """Multi-task NAM with shared feature subnetworks and task-specific outputs."""

  def __init__(
      self,
      num_inputs: int,
      num_units: int | Sequence[int],
      num_tasks: int,
      shallow: bool = True,
      feature_dropout: float = 0.0,
      dropout: float = 0.0,
      activation: str = "exu",
  ) -> None:
    super().__init__()
    self.num_inputs = num_inputs
    self.num_tasks = num_tasks
    if isinstance(num_units, int):
      self.num_units = [num_units for _ in range(num_inputs)]
    else:
      self.num_units = list(num_units)
    if len(self.num_units) != num_inputs:
      raise ValueError("The number of unit values must match the number of inputs.")
    self.shallow = shallow
    self.feature_dropout = feature_dropout
    self.dropout = dropout
    self.activation = activation
    self.feature_nns = nn.ModuleList([
        FeatureNN(
            num_units=self.num_units[index],
            dropout=dropout,
            shallow=shallow,
            feature_num=index,
            activation=activation,
            output_dim=num_tasks,
        )
        for index in range(num_inputs)
    ])
    self.bias = nn.Parameter(torch.zeros(num_tasks))

  def calc_outputs(self, x: Tensor, training: bool | None = None) -> List[Tensor]:
    """Return one output tensor per input feature with task contributions."""
    if x.ndim != 2:
      raise ValueError(f"Expected rank-2 input, got shape {tuple(x.shape)}")
    return [
        feature_nn(feature_values, training=training)
        for feature_nn, feature_values in zip(self.feature_nns, torch.split(x, 1, dim=1))
    ]

  def forward(self, x: Tensor, training: bool | None = None) -> Tensor:
    """Return the per-task logits."""
    should_train = self.training if training is None else training
    individual_outputs = self.calc_outputs(x, training=should_train)
    stacked_outputs = torch.stack(individual_outputs, dim=1)
    dropped_outputs = F.dropout(
        stacked_outputs,
        p=self.feature_dropout,
        training=should_train,
    )
    return dropped_outputs.sum(dim=1) + self.bias


class FactorizedMachine(nn.Module):
  """Second-order factorization machine interaction layer."""

  def __init__(self, input_dim: int, rank: int = 8) -> None:
    super().__init__()
    self.input_dim = input_dim
    self.rank = rank
    self.factors = nn.Parameter(torch.empty(input_dim, rank))
    self.reset_parameters()

  def reset_parameters(self) -> None:
    """Initialize latent interaction factors."""
    nn.init.xavier_uniform_(self.factors)

  def forward(self, x: Tensor) -> Tensor:
    """Return the pairwise interaction score for each sample."""
    if x.ndim != 2:
      raise ValueError(f"Expected rank-2 input, got shape {tuple(x.shape)}")
    projected = x @ self.factors
    projected_square = projected.square()
    squared_projected = (x.square()) @ self.factors.square()
    return 0.5 * (projected_square - squared_projected).sum(dim=1)


class FactorizedNAM(nn.Module):
  """NAM with an FM interaction term added directly to the output logit."""

  def __init__(
      self,
      num_inputs: int,
      num_units: int | Sequence[int],
      fm_rank: int = 8,
      shallow: bool = True,
      feature_dropout: float = 0.0,
      dropout: float = 0.0,
      activation: str = "exu",
  ) -> None:
    super().__init__()
    self.nam = NAM(
        num_inputs=num_inputs,
        num_units=num_units,
        shallow=shallow,
        feature_dropout=feature_dropout,
        dropout=dropout,
        activation=activation,
    )
    self.factorized_machine = FactorizedMachine(input_dim=num_inputs, rank=fm_rank)

  @property
  def feature_nns(self) -> nn.ModuleList:
    """Expose NAM feature networks for existing regularization utilities."""
    return self.nam.feature_nns

  def calc_outputs(self, x: Tensor, training: bool | None = None) -> List[Tensor]:
    """Return the additive NAM feature contributions."""
    return self.nam.calc_outputs(x, training=training)

  def forward(self, x: Tensor, training: bool | None = None) -> Tensor:
    """Return additive plus pairwise interaction logits."""
    additive_logits = self.nam(x, training=training)
    interaction_logits = self.factorized_machine(x)
    return additive_logits + interaction_logits


class DNN(nn.Module):
  """Deep fully connected baseline with 10 hidden layers."""

  def __init__(self, input_dim: int, dropout: float = 0.15) -> None:
    super().__init__()
    self.dropout = dropout
    self.hidden_layers = nn.ModuleList()
    last_dim = input_dim
    for _ in range(10):
      layer = nn.Linear(last_dim, 100)
      nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
      nn.init.zeros_(layer.bias)
      self.hidden_layers.append(layer)
      last_dim = 100
    self.output_layer = nn.Linear(last_dim, 1)
    nn.init.kaiming_normal_(self.output_layer.weight, nonlinearity="linear")
    nn.init.zeros_(self.output_layer.bias)

  def forward(self, x: Tensor, training: bool | None = None) -> Tensor:
    """Return the baseline prediction."""
    if x.ndim != 2:
      raise ValueError(f"Expected rank-2 input, got shape {tuple(x.shape)}")
    should_train = self.training if training is None else training
    hidden = x
    for layer in self.hidden_layers:
      hidden = F.dropout(F.relu(layer(hidden)), p=self.dropout, training=should_train)
    return self.output_layer(hidden).squeeze(-1)
