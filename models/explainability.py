"""
模型可解释性分析模块
包含SHAP分析、梯度重要性、排列重要性等功能
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from typing import Dict, List, Tuple, Optional, Callable
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("SHAP not available. Please install with: pip install shap")

import tensorflow as tf
from tensorflow.keras import Model


class ModelExplainability:
    """
    模型可解释性分析类
    提供多种模型解释方法
    """

    def __init__(
        self,
        model: Model,
        feature_names: List[str],
        sequence_length: int = 20,
        task_type: str = "classification"
    ):
        """
        初始化可解释性分析器

        参数:
            model: 训练好的Keras模型
            feature_names: 特征名称列表
            sequence_length: 序列长度
            task_type: 任务类型 ("classification" 或 "regression")
        """
        self.model = model
        self.feature_names = feature_names
        self.n_features = len(feature_names)
        self.sequence_length = sequence_length
        self.task_type = task_type
        self.explainer = None
        self.shap_values = None
        self.permutation_importance = None
        self.gradient_importance = None

    def _flatten_sequences(self, X: np.ndarray) -> np.ndarray:
        """
        将序列数据展平为 (n_samples, sequence_length * n_features)

        参数:
            X: 序列数据 (n_samples, sequence_length, n_features)

        返回:
            展平的数据
        """
        return X.reshape(X.shape[0], -1)

    def _get_feature_names_flat(self) -> List[str]:
        """
        获取展平后的特征名称（包含时间步信息）

        返回:
            特征名称列表
        """
        return [
            f"{feat}_t{t}"
            for t in range(self.sequence_length)
            for feat in self.feature_names
        ]

    def _aggregate_shap_by_feature(self, shap_values: np.ndarray) -> pd.DataFrame:
        """
        按特征聚合SHAP值（跨时间步）

        参数:
            shap_values: SHAP值数组 (n_samples, sequence_length, n_features)

        返回:
            聚合后的特征重要性 DataFrame
        """
        if len(shap_values.shape) == 2:
            # 已经是展平的
            shap_reshaped = shap_values.reshape(-1, self.sequence_length, self.n_features)
        else:
            shap_reshaped = shap_values

        # 对每个样本，在时间维度求和
        shap_by_feature = np.abs(shap_reshaped).sum(axis=1)

        # 计算平均重要性
        mean_shap = shap_by_feature.mean(axis=0)
        std_shap = shap_by_feature.std(axis=0)

        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': mean_shap,
            'std': std_shap
        }).sort_values('importance', ascending=False).reset_index(drop=True)

        return importance_df

    def explain_with_kernel_shap(
        self,
        X_explain: np.ndarray,
        X_background: Optional[np.ndarray] = None,
        n_background: int = 100,
        use_aggregated: bool = True
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        使用KernelSHAP解释模型

        参数:
            X_explain: 要解释的样本
            X_background: 背景数据集（用于SHAP）
            n_background: 如果没有提供背景数据，使用的背景样本数
            use_aggregated: 是否返回聚合后的特征重要性

        返回:
            (shap_values, importance_df)
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP is not installed. Please install with: pip install shap")

        print("Initializing KernelSHAP...")

        # 如果没有提供背景数据，从X_explain中随机采样
        if X_background is None:
            indices = np.random.choice(len(X_explain), min(n_background, len(X_explain)), replace=False)
            X_background = X_explain[indices]

        # 展平数据
        X_background_flat = self._flatten_sequences(X_background)
        X_explain_flat = self._flatten_sequences(X_explain)

        # 创建模型包装函数
        def model_predict_flat(X_flat):
            X_reshaped = X_flat.reshape(-1, self.sequence_length, self.n_features)
            return self.model.predict(X_reshaped, verbose=0).flatten()

        # 创建解释器
        self.explainer = shap.KernelExplainer(model_predict_flat, X_background_flat)

        # 计算SHAP值
        print("Calculating SHAP values...")
        self.shap_values = self.explainer.shap_values(X_explain_flat)

        # 如果是分类任务，shap_values可能是列表，取正类的SHAP值
        if isinstance(self.shap_values, list):
            # 对于二分类，取类别1的SHAP值
            if len(self.shap_values) == 2:
                self.shap_values = self.shap_values[1]

        # 聚合特征重要性
        if use_aggregated:
            importance_df = self._aggregate_shap_by_feature(self.shap_values)
        else:
            # 使用展平的特征名
            shap_flat = np.array(self.shap_values)
            importance_df = pd.DataFrame({
                'feature': self._get_feature_names_flat(),
                'importance': np.abs(shap_flat).mean(axis=0)
            }).sort_values('importance', ascending=False).reset_index(drop=True)

        print(f"SHAP analysis completed. Top feature: {importance_df.iloc[0]['feature']}")
        return self.shap_values, importance_df

    def explain_with_gradient(
        self,
        X_explain: np.ndarray,
        use_aggregated: bool = True
    ) -> pd.DataFrame:
        """
        使用梯度重要性解释模型

        参数:
            X_explain: 要解释的样本
            use_aggregated: 是否聚合时间步

        返回:
            特征重要性 DataFrame
        """
        print("Calculating gradient importance...")

        X_tensor = tf.convert_to_tensor(X_explain, dtype=tf.float32)

        with tf.GradientTape() as tape:
            tape.watch(X_tensor)
            predictions = self.model(X_tensor)

        # 计算梯度
        gradients = tape.gradient(predictions, X_tensor).numpy()

        # 计算梯度绝对值的平均值
        grad_abs = np.abs(gradients)

        if use_aggregated:
            # 按特征聚合（跨时间步）
            grad_by_feature = grad_abs.sum(axis=1)
            mean_grad = grad_by_feature.mean(axis=0)
            std_grad = grad_by_feature.std(axis=0)

            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': mean_grad,
                'std': std_grad
            }).sort_values('importance', ascending=False).reset_index(drop=True)
        else:
            # 使用展平的特征
            grad_flat = grad_abs.reshape(len(grad_abs), -1)
            mean_grad = grad_flat.mean(axis=0)

            importance_df = pd.DataFrame({
                'feature': self._get_feature_names_flat(),
                'importance': mean_grad
            }).sort_values('importance', ascending=False).reset_index(drop=True)

        self.gradient_importance = importance_df
        print(f"Gradient importance analysis completed. Top feature: {importance_df.iloc[0]['feature']}")
        return importance_df

    def calculate_permutation_importance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_repeats: int = 10,
        random_state: int = 42,
        use_aggregated: bool = True
    ) -> pd.DataFrame:
        """
        计算排列重要性

        参数:
            X: 输入数据
            y: 真实标签
            n_repeats: 排列重复次数
            random_state: 随机种子
            use_aggregated: 是否聚合时间步

        返回:
            特征重要性 DataFrame
        """
        np.random.seed(random_state)
        print(f"Calculating permutation importance with {n_repeats} repeats...")

        # 获取基准分数
        baseline_metrics = self.model.evaluate(X, y, verbose=0)
        baseline_score = baseline_metrics[1] if self.task_type == "classification" else -baseline_metrics[0]

        if use_aggregated:
            # 按特征排列（对特征的所有时间步同时排列）
            importance_scores = []
            importance_std = []

            for feat_idx in tqdm(range(self.n_features), desc="Permuting features"):
                scores = []
                for _ in range(n_repeats):
                    X_permuted = X.copy()
                    # 对该特征的所有时间步进行排列
                    X_permuted[:, :, feat_idx] = np.random.permutation(X_permuted[:, :, feat_idx].flatten()).reshape(X_permuted.shape[0], X_permuted.shape[1])
                    permuted_metrics = self.model.evaluate(X_permuted, y, verbose=0)
                    permuted_score = permuted_metrics[1] if self.task_type == "classification" else -permuted_metrics[0]
                    scores.append(baseline_score - permuted_score)

                importance_scores.append(np.mean(scores))
                importance_std.append(np.std(scores))

            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance_scores,
                'std': importance_std
            }).sort_values('importance', ascending=False).reset_index(drop=True)
        else:
            # 对每个展平的特征排列
            importance_scores = []
            feature_names_flat = self._get_feature_names_flat()

            for flat_idx in tqdm(range(len(feature_names_flat)), desc="Permuting features"):
                time_idx = flat_idx // self.n_features
                feat_idx = flat_idx % self.n_features

                scores = []
                for _ in range(n_repeats):
                    X_permuted = X.copy()
                    X_permuted[:, time_idx, feat_idx] = np.random.permutation(X_permuted[:, time_idx, feat_idx])
                    permuted_metrics = self.model.evaluate(X_permuted, y, verbose=0)
                    permuted_score = permuted_metrics[1] if self.task_type == "classification" else -permuted_metrics[0]
                    scores.append(baseline_score - permuted_score)

                importance_scores.append(np.mean(scores))

            importance_df = pd.DataFrame({
                'feature': feature_names_flat,
                'importance': importance_scores
            }).sort_values('importance', ascending=False).reset_index(drop=True)

        self.permutation_importance = importance_df
        print(f"Permutation importance completed. Top feature: {importance_df.iloc[0]['feature']}")
        return importance_df

    def evaluate_feature_stability(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        model_builder: Callable,
        n_runs: int = 5,
        random_state: int = 42
    ) -> pd.DataFrame:
        """
        评估特征重要性的稳定性

        参数:
            X_train, y_train: 训练数据
            X_val, y_val: 验证数据
            model_builder: 创建模型的函数
            n_runs: 运行次数
            random_state: 随机种子

        返回:
            特征稳定性分析结果
        """
        print(f"Evaluating feature stability with {n_runs} runs...")
        np.random.seed(random_state)

        importance_list = []

        for run_idx in range(n_runs):
            print(f"Run {run_idx + 1}/{n_runs}")

            # 重新训练模型
            model = model_builder()
            model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, verbose=0)

            # 创建临时解释器计算重要性
            temp_explainer = ModelExplainability(
                model=model,
                feature_names=self.feature_names,
                sequence_length=self.sequence_length,
                task_type=self.task_type
            )

            # 使用梯度重要性作为快速评估
            importance_df = temp_explainer.explain_with_gradient(X_val)
            importance_list.append(dict(zip(importance_df['feature'], importance_df['importance'])))

        # 分析稳定性
        importance_matrix = pd.DataFrame(importance_list)

        stability_df = pd.DataFrame({
            'feature': self.feature_names,
            'mean_importance': importance_matrix.mean().values,
            'std_importance': importance_matrix.std().values,
            'cv_importance': importance_matrix.std().values / (importance_matrix.mean().values + 1e-8)
        }).sort_values('mean_importance', ascending=False).reset_index(drop=True)

        stability_df['stability_rank'] = stability_df['cv_importance'].rank(ascending=True)

        print(f"Feature stability evaluation completed. Most stable feature: {stability_df.iloc[0]['feature']}")
        return stability_df

    def get_all_importances(
        self,
        X_explain: np.ndarray,
        X_perm: Optional[np.ndarray] = None,
        y_perm: Optional[np.ndarray] = None,
        use_shap: bool = True,
        use_gradient: bool = True,
        use_permutation: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        获取所有特征重要性方法的结果

        参数:
            X_explain: 用于解释的样本
            X_perm, y_perm: 用于排列重要性的数据
            use_shap: 是否使用SHAP
            use_gradient: 是否使用梯度重要性
            use_permutation: 是否使用排列重要性

        返回:
            包含各种重要性方法结果的字典
        """
        all_importances = {}

        if use_shap and SHAP_AVAILABLE:
            try:
                _, shap_df = self.explain_with_kernel_shap(X_explain)
                all_importances['shap'] = shap_df
            except Exception as e:
                print(f"SHAP analysis failed: {e}")

        if use_gradient:
            grad_df = self.explain_with_gradient(X_explain)
            all_importances['gradient'] = grad_df

        if use_permutation and X_perm is not None and y_perm is not None:
            perm_df = self.calculate_permutation_importance(X_perm, y_perm)
            all_importances['permutation'] = perm_df

        return all_importances
