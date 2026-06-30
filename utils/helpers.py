"""
工具函数模块
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from pathlib import Path


def plot_price_with_signals(
    df: pd.DataFrame,
    title: str = "价格走势图",
    save_path: str = None
):
    """
    绘制价格走势图（需要matplotlib）
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')

        fig, ax = plt.subplots(figsize=(15, 8))

        # 绘制价格
        ax.plot(df.index, df['close'], label='收盘价', linewidth=2)

        # 如果有MA指标，也绘制
        if 'MA20' in df.columns:
            ax.plot(df.index, df['MA20'], label='MA20', alpha=0.7)

        ax.set_title(title, fontsize=16)
        ax.set_xlabel('时间', fontsize=12)
        ax.set_ylabel('价格', fontsize=12)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

        return fig

    except ImportError:
        print("matplotlib未安装，跳过绘图")
        return None


def plot_feature_distribution(
    df: pd.DataFrame,
    feature_cols: List[str],
    save_path: str = None
):
    """
    绘制特征分布图
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import matplotlib
        matplotlib.use('Agg')

        n_cols = min(4, len(feature_cols))
        n_rows = (len(feature_cols) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        for i, col in enumerate(feature_cols):
            row = i // n_cols
            col_idx = i % n_cols
            ax = axes[row, col_idx]

            if col in df.columns:
                sns.histplot(df[col].dropna(), bins=50, ax=ax)
                ax.set_title(col, fontsize=10)
                ax.set_xlabel('')

        # 隐藏多余的子图
        for i in range(len(feature_cols), n_rows * n_cols):
            row = i // n_cols
            col_idx = i % n_cols
            axes[row, col_idx].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

        return fig

    except ImportError:
        print("matplotlib/seaborn未安装，跳过绘图")
        return None


def print_data_summary(df: pd.DataFrame, name: str = "数据集"):
    """打印数据摘要"""
    print(f"\n{'='*60}")
    print(f"{name} 摘要")
    print(f"{'='*60}")
    print(f"时间范围: {df.index[0]} 至 {df.index[-1]}")
    print(f"样本数量: {len(df)}")
    print(f"特征数量: {len(df.columns)}")
    print(f"\n前5列: {df.columns[:5].tolist()}")

    if 'close' in df.columns:
        print(f"\n价格统计:")
        print(f"  均价: {df['close'].mean():.2f}")
        print(f"  最高: {df['close'].max():.2f}")
        print(f"  最低: {df['close'].min():.2f}")

    print(f"\n缺失值统计:")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        print(missing.to_string())
    else:
        print("  无缺失值")


def save_processed_data(data_dict: Dict[str, Any], save_dir: Path, save_csv: bool = True):
    """
    保存处理后的数据

    参数:
        data_dict: 包含数据的字典
        save_dir: 保存目录
        save_csv: 是否同时保存为CSV格式
    """
    save_dir.mkdir(exist_ok=True)

    # 保存numpy数组（默认格式）
    for key, value in data_dict.items():
        if isinstance(value, np.ndarray):
            np.save(save_dir / f"{key}.npy", value)
        elif isinstance(value, pd.Index):
            pd.Series(value).to_csv(save_dir / f"{key}.csv", index=False, header=['datetime'])

    # 同时保存为CSV格式（可选）
    if save_csv:
        save_sequences_to_csv(data_dict, save_dir)

    print(f"数据已保存至: {save_dir}")


def save_sequences_to_csv(data_dict: Dict[str, Any], save_dir: Path):
    """
    将序列数据保存为CSV格式

    对于3D数组 X: (samples, timesteps, features)
        展平为2D保存: (samples, timesteps * features)
    """
    csv_dir = save_dir / "csv_format"
    csv_dir.mkdir(exist_ok=True)

    # 保存目标变量（1D数组）
    for key in ['y_train', 'y_val', 'y_test']:
        if key in data_dict and isinstance(data_dict[key], np.ndarray):
            df = pd.DataFrame(data_dict[key], columns=['target'])
            df.to_csv(csv_dir / f"{key}.csv", index=False)

    # 保存索引
    for key in ['indices_train', 'indices_val', 'indices_test']:
        if key in data_dict:
            idx = data_dict[key]
            if isinstance(idx, pd.Index):
                pd.Series(idx).to_csv(csv_dir / f"{key}.csv", index=False, header=['datetime'])

    # 保存3D输入数组 - 展平为2D
    feature_names = None
    if 'sequence_info' in data_dict:
        seq_info = data_dict['sequence_info']
        if 'feature_names' in seq_info:
            feature_names = seq_info['feature_names']

    for key in ['X_train', 'X_val', 'X_test']:
        if key in data_dict and isinstance(data_dict[key], np.ndarray):
            X = data_dict[key]
            n_samples, n_timesteps, n_features = X.shape

            # 展平为 (samples, timesteps * features)
            X_flat = X.reshape(n_samples, -1)

            # 生成列名
            if feature_names and len(feature_names) == n_features:
                columns = [f"{feat}_t{i}" for i in range(n_timesteps) for feat in feature_names]
            else:
                columns = [f"feature{f}_t{t}" for t in range(n_timesteps) for f in range(n_features)]

            df = pd.DataFrame(X_flat, columns=columns)
            df.to_csv(csv_dir / f"{key}.csv", index=False, float_format='%.8f')

    # 保存特征名称
    if feature_names:
        pd.DataFrame({'feature': feature_names}).to_csv(
            csv_dir / "feature_names.csv", index=False
        )

    print(f"  CSV格式已保存至: {csv_dir}")
