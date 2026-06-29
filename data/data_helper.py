"""
数据辅助模块
用于加载已处理的数据并准备分类任务
"""
import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import settings


def load_processed_data(data_dir: str = None):
    """
    加载已处理的数据

    参数:
        data_dir: 数据目录，默认为 output/

    返回:
        包含所有数据的字典
    """
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "output"
    else:
        data_dir = Path(data_dir)

    seq_dir = data_dir / "sequences"

    # 加载序列数据
    X_train = np.load(seq_dir / "X_train.npy")
    y_train_reg = np.load(seq_dir / "y_train.npy")
    X_val = np.load(seq_dir / "X_val.npy")
    y_val_reg = np.load(seq_dir / "y_val.npy")
    X_test = np.load(seq_dir / "X_test.npy")
    y_test_reg = np.load(seq_dir / "y_test.npy")

    # 从回归目标转换为分类目标
    print("从回归目标生成分类目标...")
    y_train = (y_train_reg > 0).astype(int)
    y_val = (y_val_reg > 0).astype(int)
    y_test = (y_test_reg > 0).astype(int)

    # 加载特征列表
    selected_features = pd.read_csv(data_dir / "selected_features.csv")['feature'].tolist()

    print(f"数据加载完成:")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}, 正样本: {y_train.sum():.0f}")
    print(f"  X_val:   {X_val.shape}, y_val:   {y_val.shape}, 正样本: {y_val.sum():.0f}")
    print(f"  X_test:  {X_test.shape}, y_test:  {y_test.shape}, 正样本: {y_test.sum():.0f}")
    print(f"  特征数: {len(selected_features)}")

    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'selected_features': selected_features,
        'input_shape': X_train.shape[1:]
    }


def normalize_data(X_train, X_val, X_test, method='standard'):
    """
    标准化数据（按特征维度）

    参数:
        X_train, X_val, X_test: 输入数据
        method: 'standard' 或 'minmax'

    返回:
        标准化后的数据
    """
    # 重塑为 (n_samples * sequence_length, n_features)
    n_train, seq_len, n_feat = X_train.shape
    X_train_reshaped = X_train.reshape(-1, n_feat)

    if method == 'standard':
        mean = X_train_reshaped.mean(axis=0)
        std = X_train_reshaped.std(axis=0)
        X_train_reshaped = (X_train_reshaped - mean) / (std + 1e-8)
        X_val_reshaped = (X_val.reshape(-1, n_feat) - mean) / (std + 1e-8)
        X_test_reshaped = (X_test.reshape(-1, n_feat) - mean) / (std + 1e-8)
    elif method == 'minmax':
        min_val = X_train_reshaped.min(axis=0)
        max_val = X_train_reshaped.max(axis=0)
        X_train_reshaped = (X_train_reshaped - min_val) / (max_val - min_val + 1e-8)
        X_val_reshaped = (X_val.reshape(-1, n_feat) - min_val) / (max_val - min_val + 1e-8)
        X_test_reshaped = (X_test.reshape(-1, n_feat) - min_val) / (max_val - min_val + 1e-8)
    else:
        raise ValueError(f"Unknown method: {method}")

    # 重塑回去
    X_train_norm = X_train_reshaped.reshape(n_train, seq_len, n_feat)
    X_val_norm = X_val_reshaped.reshape(-1, seq_len, n_feat)
    X_test_norm = X_test_reshaped.reshape(-1, seq_len, n_feat)

    return X_train_norm, X_val_norm, X_test_norm
