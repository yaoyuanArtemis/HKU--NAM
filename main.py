"""批量在多个数据集上运行基线模型对比实验。

使用下载到本地的数据集（而不是从 GCS 读取）。

注意：
    - NAM 论文使用 9 个数据集
    - 本脚本配置了 7 个数据集（6 个自动下载 + 1 个手动下载）
    - MIMIC-II 和 FICO 未包含（需要特殊授权或预处理）

Usage:
    # 只运行基线模型
    python main.py

    # 运行基线模型 + NAM
    python main.py --train_nam

    # 只运行 NAM（跳过基线）
    python main.py --only_nam
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
import pandas as pd


# 数据目录
DATA_DIR = './datasets'


# NAM 论文使用的数据集配置（使用本地文件）
DATASETS = {
    # 分类任务（sklearn 内置，自动生成）
    'BreastCancer': {
        'file': 'breast_cancer.csv',
        'task': 'classification',
        'target': 'target',
        'nam_dataset_name': 'BreastCancer'  # nam_train.py 使用的名称
    },

    # 分类任务（可从公开源下载）
    'Adult': {
        'file': 'adult.csv',
        'task': 'classification',
        'target': 'income',
        'nam_dataset_name': 'Adult'
    },
    'Heart': {
        'file': 'heart_disease.csv',
        'task': 'classification',
        'target': 'target',
        'nam_dataset_name': 'Heart'
    },
    'Telco': {
        'file': 'telco_churn.csv',
        'task': 'classification',
        'target': 'Churn',
        'nam_dataset_name': 'Telco'
    },
    'Recidivism': {
        'file': 'compas_recidivism.csv',
        'task': 'classification',
        'target': 'two_year_recid',
        'nam_dataset_name': 'Recidivism'
    },

    # 分类任务（需要手动下载）
    'Credit': {
        'file': 'creditcard.csv',
        'task': 'classification',
        'target': 'Class',
        'nam_dataset_name': 'Credit'
    },
    'Fico': {
        'file': 'heloc_dataset.csv',  # FICO HELOC 数据集
        'task': 'classification',
        'target': 'RiskPerformance',
        'nam_dataset_name': 'Fico'
    },

    # 回归任务
    'Housing': {
        'file': 'california_housing.csv',
        'task': 'regression',
        'target': 'target',
        'nam_dataset_name': 'Housing'
    },

    # 医疗数据（需要授权或自行获取）
    'Mimic2': {
        'file': 'mimic2.csv',  # 需要预处理成 CSV 格式
        'task': 'classification',
        'target': 'label',  # 原始格式中最后一列是标签
        'nam_dataset_name': 'Mimic2'
    },
}


def train_nam_on_dataset(dataset_name, config, output_dir='./all_results'):
    """训练 NAM 模型在单个数据集上。

    Args:
        dataset_name: 数据集名称
        config: 数据集配置（包含 nam_dataset_name）
        output_dir: 结果输出目录

    Returns:
        是否成功训练
    """
    print("\n" + "="*70)
    print(f"🧠 训练 NAM 模型: {dataset_name}")
    print("="*70)

    # 检查是否有 NAM 数据集名称
    if 'nam_dataset_name' not in config:
        print(f"⚠️  跳过 NAM 训练（未配置 nam_dataset_name）")
        return False

    nam_dataset_name = config['nam_dataset_name']
    regression = (config['task'] == 'regression')

    # NAM 日志目录
    nam_logdir = os.path.join(output_dir, 'nam_logs', dataset_name.lower())
    os.makedirs(nam_logdir, exist_ok=True)

    # 构建 NAM 训练命令
    cmd = [
        sys.executable, 'nam/nam_train.py',
        '--dataset_name', nam_dataset_name,
        '--training_epochs', '1000',
        '--learning_rate', '0.01',
        '--batch_size', '1024',
        '--dropout', '0.5',
        '--logdir', nam_logdir,
        '--regression', str(regression).lower(),
        '--output_regularization', '0.0',
        '--l2_regularization', '0.0',
    ]

    print(f"命令: {' '.join(cmd)}")

    try:
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        elapsed_time = time.time() - start_time

        if result.returncode == 0:
            print(f"✓ NAM 训练完成！用时: {elapsed_time:.1f}秒")

            # 保存日志
            log_file = os.path.join(nam_logdir, 'training.log')
            with open(log_file, 'w') as f:
                f.write(result.stdout)
                if result.stderr:
                    f.write("\n\n=== STDERR ===\n")
                    f.write(result.stderr)

            return True
        else:
            print(f"❌ NAM 训练失败！")
            print(f"错误: {result.stderr[:500]}")  # 只显示前 500 字符

            # 保存错误日志
            error_log = os.path.join(nam_logdir, 'error.log')
            with open(error_log, 'w') as f:
                f.write(result.stderr)
                f.write("\n\n=== STDOUT ===\n")
                f.write(result.stdout)

            return False

    except subprocess.TimeoutExpired:
        print(f"⏰ NAM 训练超时（1小时）")
        return False
    except Exception as e:
        print(f"❌ NAM 训练出错: {e}")
        return False


def run_experiment_on_dataset(dataset_name, config, output_dir='./all_results',
                              train_nam=False, only_nam=False):
    """在单个数据集上运行实验。

    Args:
        dataset_name: 数据集名称
        config: 数据集配置 (file, task, target)
        output_dir: 结果输出目录
        train_nam: 是否训练 NAM 模型
        only_nam: 是否只训练 NAM（跳过基线）

    Returns:
        是否成功运行
    """
    print("\n" + "="*70)
    print(f"📊 运行数据集: {dataset_name}")
    print("="*70)

    # 检查数据文件是否存在
    data_file = os.path.join(DATA_DIR, config['file'])

    if not os.path.exists(data_file):
        print(f"❌ 数据文件不存在: {data_file}")
        print(f"   请先运行: python download_datasets.py")
        return False

    print(f"✓ 找到数据文件: {data_file}")

    # 读取数据查看基本信息
    try:
        df = pd.read_csv(data_file)
        print(f"  数据形状: {df.shape}")
        print(f"  目标列: {config['target']}")

        if config['target'] not in df.columns:
            print(f"  ⚠️  目标列 '{config['target']}' 不存在")
            print(f"  可用列: {list(df.columns)}")
            return False

    except Exception as e:
        print(f"❌ 读取数据失败: {e}")
        return False

    baseline_success = False

    # 运行基线模型（如果不是 only_nam）
    if not only_nam:
        cmd = [
            sys.executable, 'baseline/run_experiment.py',
            '--data_path', data_file,
            '--target_column', config['target'],
            '--task', config['task'],
            '--output_dir', output_dir
        ]

        print(f"\n运行基线模型: {' '.join(cmd)}")

        try:
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            elapsed_time = time.time() - start_time

            if result.returncode == 0:
                print(f"✓ {dataset_name} 基线模型完成！用时: {elapsed_time:.1f}秒")
                baseline_success = True
            else:
                print(f"❌ {dataset_name} 基线模型失败！")
                print(f"错误信息:\n{result.stderr}")

        except subprocess.TimeoutExpired:
            print(f"⏰ {dataset_name} 基线模型超时（10分钟）")
        except Exception as e:
            print(f"❌ 基线模型运行出错: {e}")

    # 训练 NAM 模型（如果 train_nam 或 only_nam）
    nam_success = False
    if train_nam or only_nam:
        nam_success = train_nam_on_dataset(dataset_name, config, output_dir)

    # 总结
    if only_nam:
        return nam_success
    elif train_nam:
        if baseline_success and nam_success:
            print(f"\n✓ {dataset_name} 全部完成（基线 + NAM）")
        elif baseline_success:
            print(f"\n⚠️  {dataset_name} 基线完成，但 NAM 训练失败")
        return baseline_success  # 基线成功就算成功
    else:
        return baseline_success


def collect_all_results(output_dir='./all_results'):
    """收集所有数据集的结果并生成汇总报告。

    Args:
        output_dir: 结果目录

    Returns:
        汇总的 DataFrame
    """
    import glob

    print("\n" + "="*70)
    print("📊 收集所有结果...")
    print("="*70)

    result_files = glob.glob(f'{output_dir}/*_comparison.csv')

    if not result_files:
        print("未找到结果文件")
        return None

    all_results = []

    for result_file in sorted(result_files):
        dataset_name = os.path.basename(result_file).replace('_temp_comparison.csv', '')

        try:
            df = pd.read_csv(result_file)
            df['Dataset'] = dataset_name
            all_results.append(df)
            print(f"✓ {dataset_name}")
        except Exception as e:
            print(f"❌ 读取 {dataset_name} 失败: {e}")

    if not all_results:
        return None

    # 合并所有结果
    summary_df = pd.concat(all_results, ignore_index=True)

    # 重新排列列顺序
    cols = ['Dataset', 'Model'] + [col for col in summary_df.columns if col not in ['Dataset', 'Model']]
    summary_df = summary_df[cols]

    return summary_df


def generate_summary_report(summary_df, output_dir='./all_results'):
    """生成汇总报告。

    Args:
        summary_df: 汇总的结果 DataFrame
        output_dir: 输出目录
    """
    if summary_df is None or summary_df.empty:
        print("没有结果可以汇总")
        return

    print("\n" + "="*70)
    print("📄 生成汇总报告...")
    print("="*70)

    # 保存完整结果
    summary_path = os.path.join(output_dir, 'ALL_DATASETS_SUMMARY.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"✓ 完整结果: {summary_path}")

    # 生成 Markdown 报告
    report_path = os.path.join(output_dir, 'SUMMARY_REPORT.md')

    with open(report_path, 'w') as f:
        f.write("# 所有数据集模型对比汇总报告\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # 按数据集分组
        for dataset in summary_df['Dataset'].unique():
            f.write(f"## {dataset}\n\n")

            dataset_df = summary_df[summary_df['Dataset'] == dataset]
            dataset_df = dataset_df.drop(columns=['Dataset'])

            f.write(dataset_df.to_markdown(index=False))
            f.write("\n\n")

        # 总体统计
        f.write("## 总体统计\n\n")
        f.write(f"- 数据集数量: {summary_df['Dataset'].nunique()}\n")
        f.write(f"- 模型数量: {summary_df['Model'].nunique()}\n")
        f.write(f"- 总实验次数: {len(summary_df)}\n\n")

        # 最佳模型统计
        metric_cols = [col for col in summary_df.columns if 'Test' in col and ('AUROC' in col or 'RMSE' in col)]

        if metric_cols:
            metric_col = metric_cols[0]
            ascending = 'RMSE' in metric_col

            f.write(f"### 各数据集最佳模型 (按 {metric_col})\n\n")
            f.write("| Dataset | Best Model | Score |\n")
            f.write("|---------|------------|-------|\n")

            for dataset in summary_df['Dataset'].unique():
                dataset_df = summary_df[summary_df['Dataset'] == dataset]

                # Skip if all values are NaN (dataset failed)
                if dataset_df[metric_col].isna().all():
                    f.write(f"| {dataset} | N/A (failed) | N/A |\n")
                    continue

                if ascending:
                    best_idx = dataset_df[metric_col].idxmin()
                else:
                    best_idx = dataset_df[metric_col].idxmax()

                best_model = dataset_df.loc[best_idx, 'Model']
                best_score = dataset_df.loc[best_idx, metric_col]

                f.write(f"| {dataset} | {best_model} | {best_score:.4f} |\n")

    print(f"✓ Markdown 报告: {report_path}")

    # 打印简要结果
    print("\n" + "="*70)
    print("📊 简要结果")
    print("="*70)
    print(f"运行的数据集: {summary_df['Dataset'].nunique()}")
    print(f"对比的模型: {summary_df['Model'].nunique()}")
    print(f"\n数据集列表: {', '.join(sorted(summary_df['Dataset'].unique()))}")
    print(f"模型列表: {', '.join(sorted(summary_df['Model'].unique()))}")


def main():
    """主函数：批量运行所有数据集。"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='批量运行基线模型对比实验（可选 NAM）')
    parser.add_argument('--train_nam', action='store_true',
                        help='训练 NAM 模型（在基线模型之后）')
    parser.add_argument('--only_nam', action='store_true',
                        help='只训练 NAM 模型（跳过基线模型）')
    args = parser.parse_args()

    print("="*70)
    print("🚀 批量运行 NAM 基线模型对比实验")
    print("="*70)
    print(f"数据集数量: {len(DATASETS)}")
    print(f"数据集列表: {', '.join(DATASETS.keys())}")
    print(f"数据目录: {DATA_DIR}")

    if args.only_nam:
        print("模式: 只训练 NAM")
    elif args.train_nam:
        print("模式: 基线模型 + NAM")
    else:
        print("模式: 只训练基线模型")
    print("")

    # 检查数据目录
    if not os.path.exists(DATA_DIR):
        print(f"❌ 数据目录不存在: {DATA_DIR}")
        print(f"\n请先运行: python download_datasets.py")
        return

    # 创建输出目录
    output_dir = './all_results'
    os.makedirs(output_dir, exist_ok=True)

    # 记录开始时间
    start_time = time.time()

    # 运行统计
    success_count = 0
    failed_datasets = []

    # 遍历所有数据集
    for dataset_name, config in DATASETS.items():
        success = run_experiment_on_dataset(
            dataset_name, config, output_dir,
            train_nam=args.train_nam,
            only_nam=args.only_nam
        )

        if success:
            success_count += 1
        else:
            failed_datasets.append(dataset_name)

        # 短暂暂停，避免过载
        time.sleep(1)

    # 总用时
    total_time = time.time() - start_time

    # 收集和汇总结果（只在非 only_nam 模式下）
    if not args.only_nam:
        summary_df = collect_all_results(output_dir)

        if summary_df is not None:
            generate_summary_report(summary_df, output_dir)

    # 打印最终统计
    print("\n" + "="*70)
    print("✅ 批量实验完成！")
    print("="*70)
    print(f"总用时: {total_time/60:.1f} 分钟")
    print(f"成功: {success_count}/{len(DATASETS)}")

    if failed_datasets:
        print(f"失败的数据集: {', '.join(failed_datasets)}")

    print(f"\n结果保存在: {output_dir}")
    print("  - ALL_DATASETS_SUMMARY.csv  (完整结果)")
    print("  - SUMMARY_REPORT.md         (Markdown 报告)")
    print("")


if __name__ == '__main__':
    main()
