"""从公开来源下载 NAM 论文使用的数据集。

Usage:
    python download_datasets.py
"""

import os
import urllib.request
import pandas as pd
from pathlib import Path


# 数据存储目录
DATA_DIR = './datasets'


def ensure_dir(directory):
    """确保目录存在。"""
    Path(directory).mkdir(parents=True, exist_ok=True)


def download_file(url, output_path, desc=""):
    """下载文件。"""
    try:
        print(f"  下载 {desc}...")
        urllib.request.urlretrieve(url, output_path)
        print(f"  ✓ 已保存到: {output_path}")
        return True
    except Exception as e:
        print(f"  ❌ 下载失败: {e}")
        return False


def prepare_breast_cancer():
    """准备 Breast Cancer 数据集（sklearn 内置）。"""
    print("\n[1/9] Breast Cancer Wisconsin")
    print("-" * 50)

    from sklearn.datasets import load_breast_cancer

    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target

    output_path = os.path.join(DATA_DIR, 'breast_cancer.csv')
    df.to_csv(output_path, index=False)

    print(f"  ✓ 生成成功: {df.shape}")
    print(f"  ✓ 保存到: {output_path}")
    return True


def prepare_california_housing():
    """准备 California Housing 数据集（sklearn 内置）。"""
    print("\n[2/9] California Housing")
    print("-" * 50)

    from sklearn.datasets import fetch_california_housing

    data = fetch_california_housing()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target

    output_path = os.path.join(DATA_DIR, 'california_housing.csv')
    df.to_csv(output_path, index=False)

    print(f"  ✓ 生成成功: {df.shape}")
    print(f"  ✓ 保存到: {output_path}")
    return True


def download_adult():
    """下载 Adult Income 数据集。"""
    print("\n[3/9] Adult Income (Census)")
    print("-" * 50)

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    output_path = os.path.join(DATA_DIR, 'adult.data')

    if download_file(url, output_path, "Adult dataset"):
        # 添加列名
        columns = ['age', 'workclass', 'fnlwgt', 'education', 'education_num',
                  'marital_status', 'occupation', 'relationship', 'race', 'sex',
                  'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']

        try:
            df = pd.read_csv(output_path, names=columns, skipinitialspace=True)
            csv_path = os.path.join(DATA_DIR, 'adult.csv')
            df.to_csv(csv_path, index=False)
            print(f"  ✓ 已转换为 CSV: {csv_path} ({df.shape})")
            return True
        except Exception as e:
            print(f"  ⚠️  数据已下载但转换失败: {e}")
            return True

    return False


def download_heart():
    """下载 Heart Disease 数据集。"""
    print("\n[4/9] Heart Disease")
    print("-" * 50)

    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
    output_path = os.path.join(DATA_DIR, 'heart_disease.data')

    if download_file(url, output_path, "Heart Disease dataset"):
        columns = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                  'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']

        try:
            df = pd.read_csv(output_path, names=columns, na_values='?')
            df = df.dropna()  # 删除缺失值
            csv_path = os.path.join(DATA_DIR, 'heart_disease.csv')
            df.to_csv(csv_path, index=False)
            print(f"  ✓ 已转换为 CSV: {csv_path} ({df.shape})")
            return True
        except Exception as e:
            print(f"  ⚠️  数据已下载但转换失败: {e}")
            return True

    return False


def download_telco():
    """下载 Telco Customer Churn 数据集。"""
    print("\n[5/9] Telco Customer Churn")
    print("-" * 50)

    # 使用 IBM 的公开数据集
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    output_path = os.path.join(DATA_DIR, 'telco_churn.csv')

    return download_file(url, output_path, "Telco Churn dataset")


def download_credit():
    """检查 Credit Card Fraud 数据集。"""
    print("\n[6/9] Credit Card Fraud")
    print("-" * 50)

    # 检查文件是否已存在
    output_path = os.path.join(DATA_DIR, 'creditcard.csv')
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        print(f"  ✓ 文件已存在: {output_path} ({file_size:.1f} MB)")
        return True

    print("  ℹ️  Credit Card Fraud 数据集较大 (~150MB)")
    print("  ℹ️  需要从 Kaggle 手动下载:")
    print("  📎 https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
    print("  ")
    print("  下载步骤:")
    print("  1. 访问上述 Kaggle 链接")
    print("  2. 点击 'Download' 下载 creditcard.csv")
    print("  3. 将文件放到 ./datasets/ 目录")
    print("  ⏭️  跳过自动下载")
    return False


def download_recidivism():
    """下载 COMPAS Recidivism 数据集。"""
    print("\n[7/9] COMPAS Recidivism")
    print("-" * 50)

    # ProPublica 的数据
    url = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
    output_path = os.path.join(DATA_DIR, 'compas_recidivism.csv')

    return download_file(url, output_path, "COMPAS Recidivism dataset")


def download_fico():
    """检查 FICO Score 数据集。"""
    print("\n[8/9] FICO Score (HELOC)")
    print("-" * 50)

    # 检查文件是否已存在（支持多种可能的文件名）
    possible_names = ['heloc_dataset.csv', 'fico.csv', 'heloc.csv']
    for filename in possible_names:
        output_path = os.path.join(DATA_DIR, filename)
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path) / 1024  # KB
            print(f"  ✓ 文件已存在: {output_path} ({file_size:.1f} KB)")
            return True

    print("  ℹ️  FICO HELOC 数据集需要注册")
    print("  📎 https://community.fico.com/s/explainable-machine-learning-challenge")
    print("  ")
    print("  下载步骤:")
    print("  1. 访问上述链接并注册")
    print("  2. 下载 heloc_dataset_v1.csv")
    print("  3. 重命名为 heloc_dataset.csv")
    print("  4. 将文件放到 ./datasets/ 目录")
    print("  ⏭️  跳过自动下载")
    return False


def download_mimic2():
    """检查 MIMIC-II 数据集。"""
    print("\n[9/9] MIMIC-II ICU Mortality")
    print("-" * 50)

    # 检查文件是否已存在
    output_path = os.path.join(DATA_DIR, 'mimic2.csv')
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path) / 1024  # KB
        print(f"  ✓ 文件已存在: {output_path} ({file_size:.1f} KB)")
        return True

    print("  ℹ️  MIMIC-II 需要手动获取")
    print("  📎 https://mimic.mit.edu/")
    print("")
    print("  数据格式要求:")
    print("  - 文件名: mimic2.csv")
    print("  - 任务: ICU 死亡率预测（二分类）")
    print("  - 目标列名: 'label' (0=存活, 1=死亡)")
    print("  - 特征: ICU 前 48 小时生理指标")
    print("")
    print("  获取方式:")
    print("  1. 申请 PhysioNet 访问权限")
    print("  2. 联系 NAM 论文作者获取预处理数据")
    print("  3. 从其他来源获取相同预处理的数据")
    print("")
    print("  ⏭️  跳过自动下载")
    return False


def main():
    """主函数。"""
    print("="*70)
    print("📥 NAM 论文数据集下载器")
    print("="*70)
    print(f"\n数据保存目录: {DATA_DIR}")
    print("")

    # 创建数据目录
    ensure_dir(DATA_DIR)

    # 下载统计
    success_count = 0
    skip_count = 0

    # 下载各个数据集
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

    for name, func in datasets:
        try:
            result = func()
            if result:
                success_count += 1
            else:
                skip_count += 1
        except Exception as e:
            print(f"  ❌ 错误: {e}")
            skip_count += 1

    # 总结
    print("\n" + "="*70)
    print("📊 下载完成统计")
    print("="*70)
    print(f"✓ 成功: {success_count}/{len(datasets)}")
    print(f"⏭  跳过: {skip_count}/{len(datasets)}")
    print(f"\n数据保存在: {DATA_DIR}")

    # 列出已下载的文件
    print("\n已下载的数据集:")
    if os.path.exists(DATA_DIR):
        files = sorted(os.listdir(DATA_DIR))
        if files:
            for f in files:
                path = os.path.join(DATA_DIR, f)
                size = os.path.getsize(path) / 1024  # KB
                print(f"  • {f} ({size:.1f} KB)")
        else:
            print("  (无)")

    print("\n" + "="*70)
    print("💡 提示")
    print("="*70)
    print("部分数据集需要手动下载:")
    print("  • Credit Card Fraud: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
    print("  • FICO Score: https://community.fico.com/s/explainable-machine-learning-challenge")
    print("  • MIMIC-II: https://mimic.mit.edu/ (需要授权)")
    print("")


if __name__ == '__main__':
    main()
