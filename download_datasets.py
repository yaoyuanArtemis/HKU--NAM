"""Download or prepare the datasets used in the NAM study."""

import os
import urllib.request
from pathlib import Path

import pandas as pd


DATA_DIR = "./datasets"


def ensure_dir(directory):
    """Create the data directory if needed."""
    Path(directory).mkdir(parents=True, exist_ok=True)


def download_file(url, output_path, desc=""):
    """Download a file from a public URL."""
    try:
        print(f"  Downloading {desc}...")
        urllib.request.urlretrieve(url, output_path)
        print(f"  Saved to: {output_path}")
        return True
    except Exception as exc:
        print(f"  Download failed: {exc}")
        return False


def prepare_breast_cancer():
    """Prepare the sklearn breast cancer dataset."""
    print("\n[1/9] Breast Cancer Wisconsin")
    print("-" * 50)

    from sklearn.datasets import load_breast_cancer

    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target

    output_path = os.path.join(DATA_DIR, "breast_cancer.csv")
    df.to_csv(output_path, index=False)
    print(f"  Generated dataset with shape {df.shape}")
    print(f"  Saved to: {output_path}")
    return True


def prepare_california_housing():
    """Prepare the sklearn California housing dataset."""
    print("\n[2/9] California Housing")
    print("-" * 50)

    from sklearn.datasets import fetch_california_housing

    data = fetch_california_housing()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target

    output_path = os.path.join(DATA_DIR, "california_housing.csv")
    df.to_csv(output_path, index=False)
    print(f"  Generated dataset with shape {df.shape}")
    print(f"  Saved to: {output_path}")
    return True


def download_adult():
    """Download and convert the Adult Income dataset."""
    print("\n[3/9] Adult Income (Census)")
    print("-" * 50)

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    output_path = os.path.join(DATA_DIR, "adult.data")

    if download_file(url, output_path, "Adult dataset"):
        columns = [
            "age",
            "workclass",
            "fnlwgt",
            "education",
            "education_num",
            "marital_status",
            "occupation",
            "relationship",
            "race",
            "sex",
            "capital_gain",
            "capital_loss",
            "hours_per_week",
            "native_country",
            "income",
        ]

        try:
            df = pd.read_csv(output_path, names=columns, skipinitialspace=True)
            csv_path = os.path.join(DATA_DIR, "adult.csv")
            df.to_csv(csv_path, index=False)
            print(f"  Converted to CSV: {csv_path} ({df.shape})")
            return True
        except Exception as exc:
            print(f"  Download succeeded but CSV conversion failed: {exc}")
            return True

    return False


def download_heart():
    """Download and convert the Heart Disease dataset."""
    print("\n[4/9] Heart Disease")
    print("-" * 50)

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    output_path = os.path.join(DATA_DIR, "heart_disease.data")

    if download_file(url, output_path, "Heart Disease dataset"):
        columns = [
            "age",
            "sex",
            "cp",
            "trestbps",
            "chol",
            "fbs",
            "restecg",
            "thalach",
            "exang",
            "oldpeak",
            "slope",
            "ca",
            "thal",
            "target",
        ]

        try:
            df = pd.read_csv(output_path, names=columns, na_values="?")
            df = df.dropna()
            csv_path = os.path.join(DATA_DIR, "heart_disease.csv")
            df.to_csv(csv_path, index=False)
            print(f"  Converted to CSV: {csv_path} ({df.shape})")
            return True
        except Exception as exc:
            print(f"  Download succeeded but CSV conversion failed: {exc}")
            return True

    return False


def download_telco():
    """Download the Telco Customer Churn dataset."""
    print("\n[5/9] Telco Customer Churn")
    print("-" * 50)

    url = (
        "https://raw.githubusercontent.com/IBM/"
        "telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    )
    output_path = os.path.join(DATA_DIR, "telco_churn.csv")
    return download_file(url, output_path, "Telco Churn dataset")


def download_credit():
    """Check whether the Credit Card Fraud dataset is available locally."""
    print("\n[6/9] Credit Card Fraud")
    print("-" * 50)

    output_path = os.path.join(DATA_DIR, "creditcard.csv")
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"  File already present: {output_path} ({file_size:.1f} MB)")
        return True

    print("  This dataset is large (~150 MB) and must be downloaded manually.")
    print("  Source: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
    print("  Suggested procedure:")
    print("  1. Download `creditcard.csv` from Kaggle.")
    print("  2. Place the file under `./datasets/`.")
    print("  3. Re-run this script.")
    return False


def download_recidivism():
    """Download the COMPAS recidivism dataset."""
    print("\n[7/9] COMPAS Recidivism")
    print("-" * 50)

    url = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
    output_path = os.path.join(DATA_DIR, "compas_recidivism.csv")
    return download_file(url, output_path, "COMPAS Recidivism dataset")


def download_fico():
    """Check whether the FICO HELOC dataset is available locally."""
    print("\n[8/9] FICO Score (HELOC)")
    print("-" * 50)

    possible_names = ["heloc_dataset.csv", "fico.csv", "heloc.csv"]
    for filename in possible_names:
        output_path = os.path.join(DATA_DIR, filename)
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / 1024
            print(f"  File already present: {output_path} ({file_size:.1f} KB)")
            return True

    print("  The FICO HELOC dataset requires registration.")
    print("  Source: https://community.fico.com/s/explainable-machine-learning-challenge")
    print("  Suggested procedure:")
    print("  1. Register for the challenge.")
    print("  2. Download `heloc_dataset_v1.csv`.")
    print("  3. Rename it to `heloc_dataset.csv`.")
    print("  4. Place the file under `./datasets/`.")
    return False


def download_mimic2():
    """Check whether the MIMIC-II dataset is available locally."""
    print("\n[9/9] MIMIC-II ICU Mortality")
    print("-" * 50)

    output_path = os.path.join(DATA_DIR, "mimic2.csv")
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path) / 1024
        print(f"  File already present: {output_path} ({file_size:.1f} KB)")
        return True

    print("  MIMIC-II must be obtained manually.")
    print("  Source: https://mimic.mit.edu/")
    print("  Expected file format:")
    print("  - File name: `mimic2.csv`")
    print("  - Task: binary ICU mortality prediction")
    print("  - Target column: `label` (0 = survived, 1 = died)")
    print("  - Features: physiological measurements from the first 48 ICU hours")
    print("  Suggested acquisition routes:")
    print("  1. Obtain PhysioNet access approval.")
    print("  2. Request the preprocessed release from the original NAM authors.")
    print("  3. Prepare an equivalent local CSV from an approved source.")
    return False


def main():
    """Entry point for dataset preparation."""
    print("=" * 70)
    print("NAM dataset preparation utility")
    print("=" * 70)
    print(f"\nData directory: {DATA_DIR}\n")

    ensure_dir(DATA_DIR)
    success_count = 0
    skip_count = 0

    datasets = [
        ("Breast Cancer", prepare_breast_cancer),
        ("California Housing", prepare_california_housing),
        ("Adult Income", download_adult),
        ("Heart Disease", download_heart),
        ("Telco Churn", download_telco),
        ("Credit Fraud", download_credit),
        ("COMPAS Recidivism", download_recidivism),
        ("FICO Score", download_fico),
        ("MIMIC-II", download_mimic2),
    ]

    for _, func in datasets:
        try:
            result = func()
            if result:
                success_count += 1
            else:
                skip_count += 1
        except Exception as exc:
            skip_count += 1
            print(f"  Error: {exc}")

    print("\n" + "=" * 70)
    print("Preparation summary")
    print("=" * 70)
    print(f"Successful: {success_count}/{len(datasets)}")
    print(f"Skipped or unresolved: {skip_count}/{len(datasets)}")
    print(f"\nData directory: {DATA_DIR}")

    print("\nAvailable files:")
    files = sorted(os.listdir(DATA_DIR))
    if files:
        for filename in files:
            print(f"  - {filename}")
    else:
        print("  (empty)")

    print("\nNotes")
    print("  Some datasets require manual acquisition:")
    print("  - Credit Card Fraud: Kaggle")
    print("  - FICO HELOC: challenge registration")
    print("  - MIMIC-II: controlled access")


if __name__ == "__main__":
    main()
