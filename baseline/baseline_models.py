"""Baseline model comparison for NAM.

This script compares NAM against several baseline models:
- Logistic Regression / Linear Regression
- CART (Decision Tree)
- XGBoost
- DNN (MLP)
- EBM (Explainable Boosting Machine / Boosted GAM)
"""

import os
import time
from typing import Dict, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import roc_auc_score, mean_squared_error, accuracy_score
import xgboost as xgb

try:
    from interpret.glassbox import ExplainableBoostingClassifier, ExplainableBoostingRegressor
    EBM_AVAILABLE = True
except ImportError:
    EBM_AVAILABLE = False
    print("Warning: interpret package not installed. EBM models will be skipped.")
    print("Install with: pip install interpret")


class BaselineComparison:
    """Compare NAM against baseline models."""

    def __init__(self, regression: bool = False, random_state: int = 42):
        """Initialize baseline comparison.

        Args:
            regression: Whether this is a regression task (True) or classification (False)
            random_state: Random seed for reproducibility
        """
        self.regression = regression
        self.random_state = random_state
        self.models = {}
        self.results = {}

    def initialize_models(self) -> Dict[str, Any]:
        """Initialize all baseline models.

        Returns:
            Dictionary of model name to model instance
        """
        if self.regression:
            self.models = {
                'Linear': LinearRegression(),
                'CART': DecisionTreeRegressor(
                    max_depth=5,
                    min_samples_leaf=10,
                    random_state=self.random_state
                ),
                'XGBoost': xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=self.random_state,
                    tree_method='hist'
                ),
                'DNN-MLP': MLPRegressor(
                    hidden_layer_sizes=(128, 64, 32),
                    activation='relu',
                    max_iter=1000,
                    random_state=self.random_state,
                    early_stopping=True
                ),
            }

            if EBM_AVAILABLE:
                self.models['EBM'] = ExplainableBoostingRegressor(
                    random_state=self.random_state
                )
        else:
            self.models = {
                'Logistic': LogisticRegression(
                    max_iter=1000,
                    random_state=self.random_state
                ),
                'CART': DecisionTreeClassifier(
                    max_depth=5,
                    min_samples_leaf=10,
                    random_state=self.random_state
                ),
                'XGBoost': xgb.XGBClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=self.random_state,
                    tree_method='hist',
                    eval_metric='logloss'
                ),
                'DNN-MLP': MLPClassifier(
                    hidden_layer_sizes=(128, 64, 32),
                    activation='relu',
                    max_iter=1000,
                    random_state=self.random_state,
                    early_stopping=True
                ),
            }

            if EBM_AVAILABLE:
                self.models['EBM'] = ExplainableBoostingClassifier(
                    random_state=self.random_state
                )

        return self.models

    def train_and_evaluate(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> pd.DataFrame:
        """Train all models and evaluate them.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            X_test: Test features
            y_test: Test labels

        Returns:
            DataFrame with results for all models
        """
        if not self.models:
            self.initialize_models()

        results = []

        for name, model in self.models.items():
            print(f"\n{'='*50}")
            print(f"Training {name}...")
            print(f"{'='*50}")

            # Train
            start_time = time.time()

            if name == 'XGBoost':
                # Use validation set for early stopping
                if self.regression:
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        verbose=False
                    )
                else:
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        verbose=False
                    )
            else:
                model.fit(X_train, y_train)

            train_time = time.time() - start_time

            # Evaluate
            if self.regression:
                # For regression: compute RMSE
                train_pred = model.predict(X_train)
                val_pred = model.predict(X_val)
                test_pred = model.predict(X_test)

                train_metric = np.sqrt(mean_squared_error(y_train, train_pred))
                val_metric = np.sqrt(mean_squared_error(y_val, val_pred))
                test_metric = np.sqrt(mean_squared_error(y_test, test_pred))
                metric_name = 'RMSE'

            else:
                # For classification: compute AUROC
                n_classes = len(np.unique(y_train))

                if hasattr(model, 'predict_proba'):
                    train_pred_proba = model.predict_proba(X_train)
                    val_pred_proba = model.predict_proba(X_val)
                    test_pred_proba = model.predict_proba(X_test)

                    if n_classes == 2:
                        # Binary: use probability of positive class
                        train_pred = train_pred_proba[:, 1]
                        val_pred = val_pred_proba[:, 1]
                        test_pred = test_pred_proba[:, 1]
                    else:
                        # Multiclass: use all class probabilities
                        train_pred = train_pred_proba
                        val_pred = val_pred_proba
                        test_pred = test_pred_proba
                else:
                    train_pred = model.predict(X_train)
                    val_pred = model.predict(X_val)
                    test_pred = model.predict(X_test)

                # Compute AUROC
                if n_classes == 2:
                    # Binary classification
                    train_metric = roc_auc_score(y_train, train_pred)
                    val_metric = roc_auc_score(y_val, val_pred)
                    test_metric = roc_auc_score(y_test, test_pred)
                else:
                    # Multiclass: use macro average with ovr
                    train_metric = roc_auc_score(y_train, train_pred, multi_class='ovr', average='macro')
                    val_metric = roc_auc_score(y_val, val_pred, multi_class='ovr', average='macro')
                    test_metric = roc_auc_score(y_test, test_pred, multi_class='ovr', average='macro')
                metric_name = 'AUROC'

            # Store results
            result = {
                'Model': name,
                f'Train {metric_name}': train_metric,
                f'Val {metric_name}': val_metric,
                f'Test {metric_name}': test_metric,
                'Train Time (s)': train_time,
                'Num Parameters': self._count_parameters(model)
            }
            results.append(result)

            # Print results
            print(f"{name} Results:")
            print(f"  Train {metric_name}: {train_metric:.4f}")
            print(f"  Val {metric_name}: {val_metric:.4f}")
            print(f"  Test {metric_name}: {test_metric:.4f}")
            print(f"  Training Time: {train_time:.2f}s")

            self.results[name] = result

        # Create DataFrame
        results_df = pd.DataFrame(results)

        # Sort by test metric (ascending for RMSE, descending for AUROC)
        metric_col = f'Test {metric_name}'
        results_df = results_df.sort_values(
            metric_col,
            ascending=self.regression
        )

        return results_df

    def _count_parameters(self, model) -> int:
        """Count number of parameters in a model (approximation).

        Args:
            model: Trained model

        Returns:
            Approximate number of parameters
        """
        if isinstance(model, (LogisticRegression, LinearRegression)):
            if hasattr(model, 'coef_'):
                return model.coef_.size + (1 if hasattr(model, 'intercept_') else 0)
        elif isinstance(model, (DecisionTreeClassifier, DecisionTreeRegressor)):
            return model.tree_.node_count
        elif isinstance(model, (xgb.XGBClassifier, xgb.XGBRegressor)):
            # Approximate: num_trees * avg_nodes_per_tree
            return model.get_booster().num_boosted_rounds() * 31  # avg nodes
        elif isinstance(model, (MLPClassifier, MLPRegressor)):
            # Count weights in neural network
            total = 0
            for i, coef in enumerate(model.coefs_):
                total += coef.size + model.intercepts_[i].size
            return total
        elif EBM_AVAILABLE and isinstance(
            model,
            (ExplainableBoostingClassifier, ExplainableBoostingRegressor)
        ):
            # EBM stores additive shape functions
            return len(model.feature_names_in_) * 100  # approximate

        return 0

    def save_results(self, output_dir: str, dataset_name: str = 'dataset'):
        """Save comparison results to CSV.

        Args:
            output_dir: Directory to save results
            dataset_name: Name of dataset for filename
        """
        os.makedirs(output_dir, exist_ok=True)

        results_df = pd.DataFrame(list(self.results.values()))

        # Sort by test metric
        metric_name = 'RMSE' if self.regression else 'AUROC'
        metric_col = f'Test {metric_name}'
        results_df = results_df.sort_values(
            metric_col,
            ascending=self.regression
        )

        output_path = os.path.join(
            output_dir,
            f'{dataset_name}_baseline_comparison.csv'
        )
        results_df.to_csv(output_path, index=False)
        print(f"\nResults saved to: {output_path}")

        return output_path
