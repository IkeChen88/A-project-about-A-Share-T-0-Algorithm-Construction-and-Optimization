"""
时间序列构建模块
构建滚动时间窗口和历史序列作为模型输入
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional, List
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import settings


class TimeSeriesBuilder:
    """时间序列构建器"""

    def __init__(self, df: pd.DataFrame, feature_cols: List[str]):
        """
        参数:
            df: 包含特征和目标变量的DataFrame
            feature_cols: 用于构建序列的特征列
        """
        self.df = df.copy()
        self.feature_cols = feature_cols

    def create_sequences(
        self,
        sequence_length: int = None,
        target_period: int = None,
        drop_na: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, pd.Index]:
        """
        创建滚动时间窗口序列

        参数:
            sequence_length: 输入序列长度（过去N个时间步）
            target_period: 预测未来M个时间步
            drop_na: 是否删除包含NA的样本

        返回:
            X: 形状为 (n_samples, sequence_length, n_features) 的输入序列
            y: 形状为 (n_samples,) 的目标变量
            indices: 每个样本对应的时间索引
        """
        if sequence_length is None:
            sequence_length = settings.SEQUENCE_LENGTH
        if target_period is None:
            target_period = settings.TARGET_PERIOD

        # 提取特征数据
        feature_data = self.df[self.feature_cols].values
        target_data = self.df['Target_Reg'].values

        n_samples = len(self.df) - sequence_length - target_period + 1
        n_features = len(self.feature_cols)

        # 初始化数组
        X = np.zeros((n_samples, sequence_length, n_features))
        y = np.zeros(n_samples)
        indices = []

        for i in range(n_samples):
            # 输入序列：从i到i+sequence_length-1
            X[i] = feature_data[i:i + sequence_length]

            # 目标变量：i+sequence_length+target_period-1
            y[i] = target_data[i + sequence_length + target_period - 1]

            # 记录时间索引（序列结束点）
            indices.append(self.df.index[i + sequence_length - 1])

        indices = pd.Index(indices)

        if drop_na:
            # 删除包含NA的样本
            mask = ~np.isnan(X).any(axis=(1, 2)) & ~np.isnan(y)
            X = X[mask]
            y = y[mask]
            indices = indices[mask]

        print(f"创建序列: {len(X)}个样本, 输入形状={X.shape}, 目标形状={y.shape}")
        return X, y, indices

    def split_train_val_test(
        self,
        X: np.ndarray,
        y: np.ndarray,
        indices: pd.Index,
        train_ratio: float = None,
        val_ratio: float = None,
        test_ratio: float = None
    ) -> dict:
        """
        时间序列划分训练/验证/测试集

        按时间顺序划分，不打乱
        """
        if train_ratio is None:
            train_ratio = settings.TRAIN_RATIO
        if val_ratio is None:
            val_ratio = settings.VAL_RATIO
        if test_ratio is None:
            test_ratio = settings.TEST_RATIO

        n_samples = len(X)
        train_end = int(n_samples * train_ratio)
        val_end = train_end + int(n_samples * val_ratio)

        splits = {
            'X_train': X[:train_end],
            'y_train': y[:train_end],
            'indices_train': indices[:train_end],
            'X_val': X[train_end:val_end],
            'y_val': y[train_end:val_end],
            'indices_val': indices[train_end:val_end],
            'X_test': X[val_end:],
            'y_test': y[val_end:],
            'indices_test': indices[val_end:]
        }

        print(f"数据集划分: 训练集={len(splits['X_train'])}, "
              f"验证集={len(splits['X_val'])}, "
              f"测试集={len(splits['X_test'])}")

        return splits

    def create_sequences_for_prediction(
        self,
        sequence_length: int = None
    ) -> np.ndarray:
        """
        为预测创建最新的序列（用于实盘预测）
        """
        if sequence_length is None:
            sequence_length = settings.SEQUENCE_LENGTH

        feature_data = self.df[self.feature_cols].values

        # 取最后sequence_length个时间步
        X = feature_data[-sequence_length:].reshape(1, sequence_length, -1)

        return X

    def normalize_features(self, method: str = 'standard') -> pd.DataFrame:
        """
        特征标准化/归一化（可选，在创建序列之前使用）

        参数:
            method: 'standard' (标准化) 或 'minmax' (归一化)
        """
        df = self.df.copy()

        for col in self.feature_cols:
            if col not in df.columns:
                continue

            if method == 'standard':
                mean = df[col].mean()
                std = df[col].std()
                df[col] = (df[col] - mean) / (std + 1e-8)
            elif method == 'minmax':
                min_val = df[col].min()
                max_val = df[col].max()
                df[col] = (df[col] - min_val) / (max_val - min_val + 1e-8)

        self.df = df
        return df

    def get_sequence_info(self, sequence_length: int = None) -> dict:
        """获取序列信息"""
        if sequence_length is None:
            sequence_length = settings.SEQUENCE_LENGTH

        info = {
            'sequence_length': sequence_length,
            'n_features': len(self.feature_cols),
            'feature_names': self.feature_cols,
            'time_range': (self.df.index[0], self.df.index[-1]),
            'total_samples': len(self.df)
        }

        return info


def prepare_model_data(
    df: pd.DataFrame,
    selected_features: List[str],
    sequence_length: int = None,
    target_period: int = None
) -> dict:
    """
    准备模型数据的完整流水线

    参数:
        df: 包含特征和目标变量的DataFrame
        selected_features: 选中的特征列表
        sequence_length: 序列长度
        target_period: 预测周期

    返回:
        包含训练/验证/测试集的字典
    """
    builder = TimeSeriesBuilder(df, selected_features)

    # 创建序列
    X, y, indices = builder.create_sequences(
        sequence_length=sequence_length,
        target_period=target_period
    )

    # 划分数据集
    splits = builder.split_train_val_test(X, y, indices)

    # 添加序列信息
    splits['sequence_info'] = builder.get_sequence_info()

    return splits
