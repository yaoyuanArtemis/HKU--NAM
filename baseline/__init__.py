"""Baseline model utilities.

This package exposes a compact baseline suite consisting of:
- Logistic Regression / Linear Regression
- CART (Decision Tree)
- XGBoost
- DNN-MLP (Dense Neural Network)
- EBM (Explainable Boosting Machine)
"""

from baseline.baseline_models import BaselineComparison

__all__ = ["BaselineComparison"]
