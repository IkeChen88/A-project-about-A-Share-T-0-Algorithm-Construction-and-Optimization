"""
时间序列交叉验证模块
实现Walk Forward Validation（滚动验证）用于时间序列数据
"""
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Callable
from dataclasses import dataclass
from pathlib import Path
import json


@dataclass
class TimeSeriesSplit:
    """
    时间序列交叉验证分割器

    实现Walk Forward Validation:
    - 初始训练窗口
    - 滚动扩展或固定窗口
    - 验证窗口大小固定
    """
    n_splits: int = 5
    train_window_size: Optional[int] = None  # None表示扩展窗口
    val_window_size: int = None
    fixed_size: bool = False  # True=固定窗口大小, False=扩展窗口

    def __post_init__(self):
        if self.val_window_size is None:
            raise ValueError("val_window_size must be specified")

    def split(self, X, y=None, groups=None):
        """
        生成训练/验证索引

        参数:
            X: 输入数据（长度为n_samples）
            y: 目标变量（仅用于兼容）
            groups: 分组（仅用于兼容）

        产生:
            (train_idx, val_idx) 元组
        """
        n_samples = len(X)

        # 计算每个分割点
        indices = np.arange(n_samples)

        # 如果没有指定训练窗口大小，计算初始窗口
        if self.train_window_size is None:
            # 总数据 - (n_splits * val_window_size) = 初始训练窗口
            initial_train_size = n_samples - self.n_splits * self.val_window_size
            if initial_train_size <= 0:
                raise ValueError(f"Not enough data. Need at least {self.n_splits * self.val_window_size + 1} samples")
            self.train_window_size = initial_train_size

        for i in range(self.n_splits):
            # 验证集起始位置
            val_start = self.train_window_size + i * self.val_window_size
            val_end = val_start + self.val_window_size

            if val_end > n_samples:
                break

            # 训练集
            if self.fixed_size:
                # 固定窗口：从 val_start - train_window_size 到 val_start
                train_start = max(0, val_start - self.train_window_size)
                train_idx = indices[train_start:val_start]
            else:
                # 扩展窗口：从开始到 val_start
                train_idx = indices[:val_start]

            # 验证集
            val_idx = indices[val_start:val_end]

            yield train_idx, val_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        """返回分割数量"""
        return self.n_splits


class TimeSeriesCrossValidator:
    """
    时间序列交叉验证器

    支持多种验证策略：
    - Walk Forward Validation（滚动向前验证）
    - 分组验证（按天/周等）
    """

    def __init__(
        self,
        n_splits: int = 5,
        val_window_size: int = None,
        fixed_size: bool = False,
        task_type: str = "classification"
    ):
        self.n_splits = n_splits
        self.val_window_size = val_window_size
        self.fixed_size = fixed_size
        self.task_type = task_type
        self.results = []

    def cross_validate(
        self,
        model_builder: Callable,
        X: np.ndarray,
        y: np.ndarray,
        X_val_fixed: Optional[np.ndarray] = None,
        y_val_fixed: Optional[np.ndarray] = None,
        training_config=None,
        verbose: int = 1
    ) -> Dict:
        """
        执行时间序列交叉验证

        参数:
            model_builder: 创建模型的函数，返回BaseModel实例
            X, y: 完整数据集
            X_val_fixed, y_val_fixed: 固定验证集（可选，用于额外评估）
            training_config: 训练配置
            verbose: 显示详细程度

        返回:
            交叉验证结果字典
        """
        n_samples = len(X)

        # 确定验证窗口大小
        if self.val_window_size is None:
            self.val_window_size = max(1, n_samples // (self.n_splits + 2))

        # 创建分割器
        splitter = TimeSeriesSplit(
            n_splits=self.n_splits,
            val_window_size=self.val_window_size,
            fixed_size=self.fixed_size
        )

        fold_results = []

        if verbose > 0:
            print(f"\n{'='*60}")
            print(f"时间序列交叉验证")
            print(f"{'='*60}")
            print(f"分割数: {self.n_splits}")
            print(f"验证窗口大小: {self.val_window_size}")
            print(f"窗口类型: {'固定' if self.fixed_size else '扩展'}")
            print(f"总样本数: {n_samples}")

        for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(X)):
            if verbose > 0:
                print(f"\n--- Fold {fold_idx + 1}/{self.n_splits} ---")
                print(f"训练: {len(train_idx)} 样本 ({train_idx[0]}:{train_idx[-1]})")
                print(f"验证: {len(val_idx)} 样本 ({val_idx[0]}:{val_idx[-1]})")

            # 准备数据
            X_train_fold, y_train_fold = X[train_idx], y[train_idx]
            X_val_fold, y_val_fold = X[val_idx], y[val_idx]

            # 创建并训练模型
            model = model_builder()
            history = model.train(
                X_train_fold, y_train_fold,
                X_val_fold, y_val_fold,
                training_config=training_config,
                verbose=verbose
            )

            # 评估
            val_metrics = model.evaluate(X_val_fold, y_val_fold)

            # 在固定验证集上评估（如果提供）
            fixed_metrics = None
            if X_val_fixed is not None and y_val_fixed is not None:
                fixed_metrics = model.evaluate(X_val_fixed, y_val_fixed)

            # 保存结果
            result = {
                'fold': fold_idx + 1,
                'train_size': len(train_idx),
                'val_size': len(val_idx),
                'train_indices': (int(train_idx[0]), int(train_idx[-1])),
                'val_indices': (int(val_idx[0]), int(val_idx[-1])),
                'val_metrics': val_metrics,
                'fixed_val_metrics': fixed_metrics,
                'history': history
            }
            fold_results.append(result)

            if verbose > 0:
                print(f"验证结果: {val_metrics}")

        # 汇总结果
        self.results = fold_results
        summary = self._summarize_results(fold_results)

        if verbose > 0:
            print(f"\n{'='*60}")
            print(f"交叉验证汇总")
            print(f"{'='*60}")
            for key, value in summary.items():
                print(f"{key}: {value}")

        return {
            'fold_results': fold_results,
            'summary': summary
        }

    def _find_metric_key(self, metrics: Dict, target: str) -> str:
        """在metrics字典中灵活查找指标key（处理TF可能的后缀）"""
        # 精确匹配
        if target in metrics:
            return target
        # 查找以target开头的key
        for key in metrics:
            if key == target or key.startswith(f'{target}_'):
                return key
        # 回退：返回原target
        return target

    def _summarize_results(self, fold_results: List[Dict]) -> Dict:
        """汇总各折的结果"""
        # 收集所有指标
        metric_names = fold_results[0]['val_metrics'].keys()
        summary = {}

        for metric in metric_names:
            values = [r['val_metrics'][metric] for r in fold_results]
            summary[f'{metric}_mean'] = float(np.mean(values))
            summary[f'{metric}_std'] = float(np.std(values))
            summary[f'{metric}_values'] = [float(v) for v in values]

        # 最佳折 — 使用灵活key查找
        if self.task_type == 'classification':
            auc_key = self._find_metric_key(fold_results[0]['val_metrics'], 'auc')
            best_idx = int(np.argmax([r['val_metrics'][auc_key] for r in fold_results]))
        else:
            loss_key = self._find_metric_key(fold_results[0]['val_metrics'], 'loss')
            best_idx = int(np.argmin([r['val_metrics'][loss_key] for r in fold_results]))

        summary['best_fold'] = best_idx + 1
        summary['best_metrics'] = fold_results[best_idx]['val_metrics']

        return summary

    def save_results(self, filepath: str):
        """保存交叉验证结果"""
        save_path = Path(filepath)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # 转换为可序列化格式
        serializable = []
        for r in self.results:
            r_copy = r.copy()
            if 'history' in r_copy:
                # 转换history中的numpy数组
                hist = {}
                for k, v in r_copy['history'].items():
                    if isinstance(v, np.ndarray):
                        hist[k] = [float(x) for x in v]
                    else:
                        hist[k] = v
                r_copy['history'] = hist
            serializable.append(r_copy)

        with open(save_path, 'w') as f:
            json.dump(serializable, f, indent=2)

    def get_results(self):
        """获取结果"""
        return self.results


def blocked_time_series_split(
    X,
    n_blocks: int = 10,
    val_blocks: int = 1,
    gap_blocks: int = 0
):
    """
    分块时间序列分割

    参数:
        X: 输入数据
        n_blocks: 总块数
        val_blocks: 验证块数
        gap_blocks: 训练和验证之间的间隔块数

    产生:
        (train_idx, val_idx) 元组
    """
    n_samples = len(X)
    block_size = n_samples // n_blocks

    indices = np.arange(n_samples)

    for i in range(n_blocks - val_blocks - gap_blocks):
        # 训练块：0 到 i
        train_end = (i + 1) * block_size
        train_idx = indices[:train_end]

        # 间隔跳过
        val_start = train_end + gap_blocks * block_size

        # 验证块
        val_end = val_start + val_blocks * block_size
        val_idx = indices[val_start:val_end]

        if len(val_idx) == 0:
            break

        yield train_idx, val_idx
