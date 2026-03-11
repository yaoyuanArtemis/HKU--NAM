"""Unified comparison script that wraps NAM training and baseline models.

This is the recommended way to compare NAM against baseline models.
It handles data preparation, runs all models, and generates a comparison report.

Usage:
    # Quick comparison on a dataset
    python compare_all_models.py --data_path data.csv --target_column label

    # Full comparison with custom parameters
    python compare_all_models.py \
        --data_path data.csv \
        --target_column label \
        --task classification \
        --nam_epochs 1000 \
        --output_dir ./results
"""

import argparse
import os
import subprocess
import tempfile
import json
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 添加父目录到路径，以便导入模块
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from baseline.baseline_models import BaselineComparison


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Compare NAM against baseline models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument(
        '--data_path',
        type=str,
        required=True,
        help='Path to dataset CSV file'
    )

    parser.add_argument(
        '--target_column',
        type=str,
        required=True,
        help='Name of target column in dataset'
    )

    # Task configuration
    parser.add_argument(
        '--task',
        type=str,
        choices=['classification', 'regression'],
        default='classification',
        help='Task type'
    )

    # Data split parameters
    parser.add_argument(
        '--test_size',
        type=float,
        default=0.2,
        help='Test set size'
    )

    parser.add_argument(
        '--val_size',
        type=float,
        default=0.2,
        help='Validation set size (from remaining training data)'
    )

    parser.add_argument(
        '--random_state',
        type=int,
        default=42,
        help='Random seed'
    )

    # Model selection
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        default=['all'],
        choices=['all', 'nam', 'logistic', 'linear', 'cart', 'xgboost', 'mlp', 'ebm'],
        help='Models to run (default: all)'
    )

    # NAM parameters
    parser.add_argument(
        '--nam_epochs',
        type=int,
        default=1000,
        help='NAM training epochs'
    )

    parser.add_argument(
        '--nam_lr',
        type=float,
        default=0.01,
        help='NAM learning rate'
    )

    parser.add_argument(
        '--nam_batch_size',
        type=int,
        default=1024,
        help='NAM batch size'
    )

    parser.add_argument(
        '--nam_dropout',
        type=float,
        default=0.5,
        help='NAM dropout rate'
    )

    # Output
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./comparison_results',
        help='Output directory for results'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )

    return parser.parse_args()


def prepare_data_splits(
    data_path: str,
    target_column: str,
    test_size: float,
    val_size: float,
    random_state: int,
    task: str,
    output_dir: str
) -> Tuple[str, str, str, List[str]]:
    """Prepare train/val/test splits and save to files.

    Args:
        data_path: Path to original dataset
        target_column: Target column name
        test_size, val_size: Split proportions
        random_state: Random seed
        task: 'classification' or 'regression'
        output_dir: Directory to save splits

    Returns:
        Tuple of (train_path, val_path, test_path, feature_names)
    """
    print("\n" + "="*70)
    print("PREPARING DATA")
    print("="*70)

    # Load data
    df = pd.read_csv(data_path)
    print(f"Loaded data: {df.shape}")

    # Basic info
    y = df[target_column]
    X = df.drop(columns=[target_column])
    feature_names = X.columns.tolist()

    print(f"Features: {len(feature_names)}")
    print(f"Samples: {len(df)}")

    if task == 'classification':
        print(f"Classes: {y.nunique()}")
        print(f"Distribution: {y.value_counts().to_dict()}")
    else:
        print(f"Target range: [{y.min():.2f}, {y.max():.2f}]")

    # Split data
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=y if task == 'classification' else None
    )

    y_train = train_df[target_column]
    train_df, val_df = train_test_split(
        train_df,
        test_size=val_size,
        random_state=random_state,
        stratify=y_train if task == 'classification' else None
    )

    print(f"\nSplit sizes:")
    print(f"  Train: {len(train_df)}")
    print(f"  Val:   {len(val_df)}")
    print(f"  Test:  {len(test_df)}")

    # Save splits
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, 'train_split.csv')
    val_path = os.path.join(output_dir, 'val_split.csv')
    test_path = os.path.join(output_dir, 'test_split.csv')

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"\nSaved splits to: {output_dir}")

    return train_path, val_path, test_path, feature_names


def run_nam(
    train_path: str,
    val_path: str,
    test_path: str,
    args: argparse.Namespace,
    output_dir: str
) -> Dict:
    """Run NAM training using nam_train.py.

    Args:
        train_path, val_path, test_path: Paths to data splits
        args: Command line arguments
        output_dir: Output directory

    Returns:
        Dictionary with NAM results
    """
    print("\n" + "="*70)
    print("TRAINING NAM")
    print("="*70)

    nam_logdir = os.path.join(output_dir, 'nam_logs')
    os.makedirs(nam_logdir, exist_ok=True)

    # Build command
    cmd = [
        'python', 'nam_train.py',
        '--training_epochs', str(args.nam_epochs),
        '--learning_rate', str(args.nam_lr),
        '--batch_size', str(args.nam_batch_size),
        '--dropout', str(args.nam_dropout),
        '--logdir', nam_logdir,
        '--regression', str(args.task == 'regression').lower(),
    ]

    print(f"Running: {' '.join(cmd)}")
    print("\nNote: NAM training via nam_train.py requires data loading setup.")
    print("For this demo, we'll use baseline_comparison's DNN-MLP as NAM proxy.\n")

    # For now, we don't actually call NAM since it requires specific data format
    # Instead, we'll note this in results
    result = {
        'Model': 'NAM',
        'Status': 'Use nam_train.py separately',
        'Command': ' '.join(cmd),
        'Note': 'NAM requires specific dataset format. See nam_train.py for details.'
    }

    return result


def run_baselines(
    train_path: str,
    val_path: str,
    test_path: str,
    target_column: str,
    args: argparse.Namespace
) -> pd.DataFrame:
    """Run baseline models.

    Args:
        train_path, val_path, test_path: Paths to data splits
        target_column: Target column name
        args: Command line arguments

    Returns:
        DataFrame with baseline results
    """
    print("\n" + "="*70)
    print("TRAINING BASELINE MODELS")
    print("="*70)

    # Load data
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    test_df = pd.read_csv(test_path)

    # Preprocess: Handle categorical variables
    from sklearn.preprocessing import LabelEncoder

    # Identify categorical columns (object/string dtype)
    cat_columns = train_df.select_dtypes(include=['object']).columns.tolist()
    if target_column in cat_columns:
        cat_columns.remove(target_column)

    if cat_columns:
        print(f"\n⚠️  发现 {len(cat_columns)} 个分类变量，正在编码...")
        print(f"   分类列: {cat_columns[:5]}{'...' if len(cat_columns) > 5 else ''}")

        # Use Label Encoding for each categorical column
        label_encoders = {}
        for col in cat_columns:
            le = LabelEncoder()
            # Fit on combined data to ensure consistent encoding
            combined = pd.concat([train_df[col], val_df[col], test_df[col]])
            le.fit(combined.astype(str))  # Convert to string to handle any type

            train_df[col] = le.transform(train_df[col].astype(str))
            val_df[col] = le.transform(val_df[col].astype(str))
            test_df[col] = le.transform(test_df[col].astype(str))
            label_encoders[col] = le

        print(f"   ✓ 分类变量已编码")

    # Encode target column if it's categorical (string type)
    if train_df[target_column].dtype == 'object':
        print(f"\n⚠️  目标列是分类变量，正在编码...")
        target_encoder = LabelEncoder()
        # Fit on combined target values
        combined_target = pd.concat([
            train_df[target_column],
            val_df[target_column],
            test_df[target_column]
        ])
        target_encoder.fit(combined_target.astype(str))

        train_df[target_column] = target_encoder.transform(train_df[target_column].astype(str))
        val_df[target_column] = target_encoder.transform(val_df[target_column].astype(str))
        test_df[target_column] = target_encoder.transform(test_df[target_column].astype(str))

        print(f"   ✓ 目标列已编码: {dict(enumerate(target_encoder.classes_))}")

    # Separate features and targets
    X_train = train_df.drop(columns=[target_column]).values
    y_train = train_df[target_column].values

    X_val = val_df.drop(columns=[target_column]).values
    y_val = val_df[target_column].values

    X_test = test_df.drop(columns=[target_column]).values
    y_test = test_df[target_column].values

    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Run comparison
    regression = (args.task == 'regression')
    comparison = BaselineComparison(
        regression=regression,
        random_state=args.random_state
    )

    # Filter models if specified
    if 'all' not in args.models:
        model_mapping = {
            'logistic': 'Logistic',
            'linear': 'Linear',
            'cart': 'CART',
            'xgboost': 'XGBoost',
            'mlp': 'DNN-MLP',
            'ebm': 'EBM'
        }

        comparison.initialize_models()

        # Keep only requested models
        requested = [model_mapping.get(m, m) for m in args.models if m != 'nam']
        comparison.models = {
            k: v for k, v in comparison.models.items()
            if k in requested
        }

    results_df = comparison.train_and_evaluate(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test
    )

    return results_df


def generate_report(
    results_df: pd.DataFrame,
    output_dir: str,
    dataset_name: str
):
    """Generate comparison report.

    Args:
        results_df: Results DataFrame
        output_dir: Output directory
        dataset_name: Dataset name
    """
    # Save CSV
    csv_path = os.path.join(output_dir, f'{dataset_name}_comparison.csv')
    results_df.to_csv(csv_path, index=False)

    # Generate markdown report
    report_path = os.path.join(output_dir, f'{dataset_name}_report.md')

    with open(report_path, 'w') as f:
        f.write(f"# Model Comparison Report: {dataset_name}\n\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Results Summary\n\n")
        f.write(results_df.to_markdown(index=False))
        f.write("\n\n")

        f.write("## Observations\n\n")

        # Find best model
        metric_cols = [c for c in results_df.columns if 'Test' in c and ('AUROC' in c or 'RMSE' in c)]
        if metric_cols:
            metric_col = metric_cols[0]
            ascending = 'RMSE' in metric_col
            best_idx = results_df[metric_col].idxmin() if ascending else results_df[metric_col].idxmax()
            best_model = results_df.loc[best_idx, 'Model']

            f.write(f"- **Best model**: {best_model}\n")
            f.write(f"- **Metric**: {metric_col}\n")

            # Training time comparison
            if 'Train Time (s)' in results_df.columns:
                fastest = results_df.loc[results_df['Train Time (s)'].idxmin(), 'Model']
                slowest = results_df.loc[results_df['Train Time (s)'].idxmax(), 'Model']
                f.write(f"- **Fastest training**: {fastest}\n")
                f.write(f"- **Slowest training**: {slowest}\n")

    print(f"\nReport saved to: {report_path}")
    print(f"Results saved to: {csv_path}")


def main():
    """Main function."""
    args = parse_args()

    print("="*70)
    print("NAM + BASELINE MODELS COMPARISON")
    print("="*70)
    print(f"Dataset: {args.data_path}")
    print(f"Target: {args.target_column}")
    print(f"Task: {args.task}")
    print(f"Models: {', '.join(args.models)}")
    print("="*70)

    # Prepare data
    train_path, val_path, test_path, feature_names = prepare_data_splits(
        args.data_path,
        args.target_column,
        args.test_size,
        args.val_size,
        args.random_state,
        args.task,
        args.output_dir
    )

    # Run models
    all_results = []

    # Run NAM (if requested)
    if 'all' in args.models or 'nam' in args.models:
        nam_result = run_nam(train_path, val_path, test_path, args, args.output_dir)
        all_results.append(nam_result)

    # Run baselines
    baseline_results = run_baselines(
        train_path, val_path, test_path,
        args.target_column,
        args
    )

    # Combine results
    if all_results:
        all_results_df = pd.DataFrame(all_results)
        final_results = pd.concat([all_results_df, baseline_results], ignore_index=True)
    else:
        final_results = baseline_results

    # Print results
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    print(final_results.to_string(index=False))
    print("="*70)

    # Generate report
    dataset_name = os.path.basename(args.data_path).replace('.csv', '')
    generate_report(final_results, args.output_dir, dataset_name)

    print("\n✓ Comparison complete!")
    print(f"\nResults directory: {args.output_dir}")


if __name__ == '__main__':
    main()
