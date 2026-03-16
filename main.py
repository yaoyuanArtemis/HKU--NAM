"""Batch runner for baseline and NAM experiments across multiple datasets.

This script operates on locally prepared datasets rather than the original
cloud-hosted NAM assets.

Usage:
    # Baselines only
    python main.py

    # Baselines followed by NAM
    python main.py --train_nam

    # NAM only
    python main.py --only_nam
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime

import pandas as pd


DATA_DIR = "./datasets"


DATASETS = {
    "BreastCancer": {
        "file": "breast_cancer.csv",
        "task": "classification",
        "target": "target",
        "nam_dataset_name": "BreastCancer",
    },
    "Adult": {
        "file": "adult.csv",
        "task": "classification",
        "target": "income",
        "nam_dataset_name": "Adult",
    },
    "Heart": {
        "file": "heart_disease.csv",
        "task": "classification",
        "target": "target",
        "nam_dataset_name": "Heart",
    },
    "Telco": {
        "file": "telco_churn.csv",
        "task": "classification",
        "target": "Churn",
        "nam_dataset_name": "Telco",
    },
    "Recidivism": {
        "file": "compas_recidivism.csv",
        "task": "classification",
        "target": "two_year_recid",
        "nam_dataset_name": "Recidivism",
    },
    "Credit": {
        "file": "creditcard.csv",
        "task": "classification",
        "target": "Class",
        "nam_dataset_name": "Credit",
    },
    "Fico": {
        "file": "heloc_dataset.csv",
        "task": "classification",
        "target": "RiskPerformance",
        "nam_dataset_name": "Fico",
    },
    "Housing": {
        "file": "california_housing.csv",
        "task": "regression",
        "target": "target",
        "nam_dataset_name": "Housing",
    },
    "Mimic2": {
        "file": "mimic2.csv",
        "task": "classification",
        "target": "label",
        "nam_dataset_name": "Mimic2",
    },
}


def train_nam_on_dataset(dataset_name, config, output_dir="./all_results"):
    """Train NAM on a single configured dataset."""
    print("\n" + "=" * 70)
    print(f"Training NAM on {dataset_name}")
    print("=" * 70)

    if "nam_dataset_name" not in config:
        print("Warning: skipping NAM because `nam_dataset_name` is not configured.")
        return False

    nam_dataset_name = config["nam_dataset_name"]
    regression = config["task"] == "regression"
    nam_logdir = os.path.join(output_dir, "nam_logs", dataset_name.lower())
    os.makedirs(nam_logdir, exist_ok=True)

    cmd = [
        sys.executable,
        "nam_train.py",
        "--dataset_name",
        nam_dataset_name,
        "--training_epochs",
        "1000",
        "--learning_rate",
        "0.01",
        "--batch_size",
        "1024",
        "--dropout",
        "0.5",
        "--logdir",
        nam_logdir,
        "--regression",
        str(regression).lower(),
        "--output_regularization",
        "0.0",
        "--l2_regularization",
        "0.0",
    ]

    print(f"Command: {' '.join(cmd)}")

    try:
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        elapsed_time = time.time() - start_time

        if result.returncode == 0:
            print(f"NAM training completed in {elapsed_time:.1f}s.")
            log_file = os.path.join(nam_logdir, "training.log")
            with open(log_file, "w", encoding="utf-8") as handle:
                handle.write(result.stdout)
                if result.stderr:
                    handle.write("\n\n=== STDERR ===\n")
                    handle.write(result.stderr)
            return True

        print("NAM training failed.")
        print(f"Error excerpt: {result.stderr[:500]}")
        error_log = os.path.join(nam_logdir, "error.log")
        with open(error_log, "w", encoding="utf-8") as handle:
            handle.write(result.stderr)
            handle.write("\n\n=== STDOUT ===\n")
            handle.write(result.stdout)
        return False

    except subprocess.TimeoutExpired:
        print("NAM training timed out after one hour.")
        return False
    except Exception as exc:
        print(f"NAM training raised an exception: {exc}")
        return False


def run_experiment_on_dataset(
    dataset_name,
    config,
    output_dir="./all_results",
    train_nam=False,
    only_nam=False,
):
    """Run the configured workflow on a single dataset."""
    print("\n" + "=" * 70)
    print(f"Running dataset: {dataset_name}")
    print("=" * 70)

    data_file = os.path.join(DATA_DIR, config["file"])
    if not os.path.exists(data_file):
        print(f"Missing dataset file: {data_file}")
        print("Run `python download_datasets.py` first.")
        return False

    print(f"Found dataset file: {data_file}")

    try:
        df = pd.read_csv(data_file)
        print(f"Data shape: {df.shape}")
        print(f"Target column: {config['target']}")
        if config["target"] not in df.columns:
            print(f"Configured target column is absent: {config['target']}")
            print(f"Available columns: {list(df.columns)}")
            return False
    except Exception as exc:
        print(f"Failed to read dataset: {exc}")
        return False

    baseline_success = False
    if not only_nam:
        cmd = [
            sys.executable,
            "baseline/run_experiment.py",
            "--data_path",
            data_file,
            "--target_column",
            config["target"],
            "--task",
            config["task"],
            "--output_dir",
            output_dir,
        ]

        print(f"\nRunning baselines: {' '.join(cmd)}")

        try:
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            elapsed_time = time.time() - start_time

            if result.returncode == 0:
                print(f"Baseline run for {dataset_name} completed in {elapsed_time:.1f}s.")
                baseline_success = True
            else:
                print(f"Baseline run for {dataset_name} failed.")
                print(f"stderr:\n{result.stderr}")
        except subprocess.TimeoutExpired:
            print(f"Baseline run for {dataset_name} timed out after ten minutes.")
        except Exception as exc:
            print(f"Baseline run raised an exception: {exc}")

    nam_success = False
    if train_nam or only_nam:
        nam_success = train_nam_on_dataset(dataset_name, config, output_dir)

    if only_nam:
        return nam_success
    if train_nam:
        if baseline_success and nam_success:
            print(f"{dataset_name}: baselines and NAM completed.")
        elif baseline_success:
            print(f"{dataset_name}: baselines completed, NAM failed.")
        return baseline_success
    return baseline_success


def collect_all_results(output_dir="./all_results"):
    """Collect per-dataset result tables into one summary table."""
    import glob

    print("\n" + "=" * 70)
    print("Collecting result files")
    print("=" * 70)

    result_files = glob.glob(f"{output_dir}/*_comparison.csv")
    if not result_files:
        print("No comparison files were found.")
        return None

    all_results = []
    for result_file in sorted(result_files):
        dataset_name = os.path.basename(result_file).replace("_temp_comparison.csv", "")
        try:
            df = pd.read_csv(result_file)
            df["Dataset"] = dataset_name
            all_results.append(df)
            print(f"Loaded {dataset_name}")
        except Exception as exc:
            print(f"Failed to load {dataset_name}: {exc}")

    if not all_results:
        return None

    summary_df = pd.concat(all_results, ignore_index=True)
    cols = ["Dataset", "Model"] + [
        col for col in summary_df.columns if col not in ["Dataset", "Model"]
    ]
    return summary_df[cols]


def generate_summary_report(summary_df, output_dir="./all_results"):
    """Write a consolidated CSV and markdown report."""
    if summary_df is None or summary_df.empty:
        print("No results are available for summarization.")
        return

    print("\n" + "=" * 70)
    print("Writing summary report")
    print("=" * 70)

    summary_path = os.path.join(output_dir, "ALL_DATASETS_SUMMARY.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Saved consolidated table: {summary_path}")

    report_path = os.path.join(output_dir, "SUMMARY_REPORT.md")
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("# Consolidated Model Comparison Report\n\n")
        handle.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        for dataset in summary_df["Dataset"].unique():
            handle.write(f"## {dataset}\n\n")
            dataset_df = summary_df[summary_df["Dataset"] == dataset].drop(columns=["Dataset"])
            handle.write(dataset_df.to_markdown(index=False))
            handle.write("\n\n")

        handle.write("## Aggregate Statistics\n\n")
        handle.write(f"- Number of datasets: {summary_df['Dataset'].nunique()}\n")
        handle.write(f"- Number of models: {summary_df['Model'].nunique()}\n")
        handle.write(f"- Number of experimental records: {len(summary_df)}\n\n")

        metric_cols = [
            col
            for col in summary_df.columns
            if "Test" in col and ("AUROC" in col or "RMSE" in col)
        ]
        if metric_cols:
            metric_col = metric_cols[0]
            ascending = "RMSE" in metric_col

            handle.write(f"### Best Model per Dataset ({metric_col})\n\n")
            handle.write("| Dataset | Best Model | Score |\n")
            handle.write("|---------|------------|-------|\n")

            for dataset in summary_df["Dataset"].unique():
                dataset_df = summary_df[summary_df["Dataset"] == dataset]
                if dataset_df[metric_col].isna().all():
                    handle.write(f"| {dataset} | N/A | N/A |\n")
                    continue

                best_idx = (
                    dataset_df[metric_col].idxmin()
                    if ascending
                    else dataset_df[metric_col].idxmax()
                )
                best_model = dataset_df.loc[best_idx, "Model"]
                best_score = dataset_df.loc[best_idx, metric_col]
                handle.write(f"| {dataset} | {best_model} | {best_score:.4f} |\n")

    print(f"Saved markdown report: {report_path}")

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Datasets processed: {summary_df['Dataset'].nunique()}")
    print(f"Models compared: {summary_df['Model'].nunique()}")
    print(f"\nDatasets: {', '.join(sorted(summary_df['Dataset'].unique()))}")
    print(f"Models: {', '.join(sorted(summary_df['Model'].unique()))}")


def main():
    """Entry point for the batch workflow."""
    parser = argparse.ArgumentParser(
        description="Run batched baseline comparisons with optional NAM training."
    )
    parser.add_argument(
        "--train_nam",
        action="store_true",
        help="Train NAM after the baseline runs.",
    )
    parser.add_argument(
        "--only_nam",
        action="store_true",
        help="Run NAM only and skip the baselines.",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Batched NAM and baseline workflow")
    print("=" * 70)
    print(f"Number of datasets: {len(DATASETS)}")
    print(f"Datasets: {', '.join(DATASETS.keys())}")
    print(f"Data directory: {DATA_DIR}")

    if args.only_nam:
        print("Mode: NAM only")
    elif args.train_nam:
        print("Mode: baselines followed by NAM")
    else:
        print("Mode: baselines only")
    print("")

    if not os.path.exists(DATA_DIR):
        print(f"Missing data directory: {DATA_DIR}")
        print("Run `python download_datasets.py` first.")
        return

    output_dir = "./all_results"
    os.makedirs(output_dir, exist_ok=True)

    start_time = time.time()
    success_count = 0
    failed_datasets = []

    for dataset_name, config in DATASETS.items():
        success = run_experiment_on_dataset(
            dataset_name,
            config,
            output_dir,
            train_nam=args.train_nam,
            only_nam=args.only_nam,
        )
        if success:
            success_count += 1
        else:
            failed_datasets.append(dataset_name)
        time.sleep(1)

    total_time = time.time() - start_time

    if not args.only_nam:
        summary_df = collect_all_results(output_dir)
        if summary_df is not None:
            generate_summary_report(summary_df, output_dir)

    print("\n" + "=" * 70)
    print("Workflow complete")
    print("=" * 70)
    print(f"Total runtime: {total_time / 60:.1f} minutes")
    print(f"Successful datasets: {success_count}/{len(DATASETS)}")
    if failed_datasets:
        print(f"Failed datasets: {', '.join(failed_datasets)}")

    print(f"\nOutputs written to: {output_dir}")
    print("  - ALL_DATASETS_SUMMARY.csv")
    print("  - SUMMARY_REPORT.md")


if __name__ == "__main__":
    main()
