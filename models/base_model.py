"""
基础模型类
包含通用训练、评估、保存/加载功能
支持二分类任务（BCE损失 + AUC指标）
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
    TensorBoard
)
from tensorflow.keras.metrics import AUC, Precision, Recall
import os
from pathlib import Path
from typing import Dict, Optional
import json

from config.model_config import ModelConfig, TrainingConfig


class BaseModel:
    """基础模型类 - 支持二分类任务"""

    def __init__(self, config: ModelConfig, name: str = "base_model", task_type: str = "classification"):
        self.config = config
        self.name = name
        self.task_type = task_type  # "classification" 或 "regression"
        self.model: Optional[Model] = None
        self.history = None
        self.is_compiled = False

    def build_model(self):
        raise NotImplementedError("Subclasses must implement build_model")

    def compile(self, optimizer: Optional[tf.keras.optimizers.Optimizer] = None, learning_rate: float = None):
        """
        编译模型

        参数:
            optimizer: 优化器实例（可选）
            learning_rate: 学习率（如果提供，会覆盖config中的值）
        """
        if self.model is None:
            self.build_model()

        if optimizer is None:
            lr = learning_rate if learning_rate is not None else self.config.learning_rate
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        if self.task_type == "classification":
            # 二分类任务
            self.model.compile(
                optimizer=optimizer,
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=[
                    AUC(name='auc'),
                    Precision(name='precision'),
                    Recall(name='recall'),
                    'accuracy'
                ]
            )
            # 显式存储指标名（model.metrics_names 可能有 TF 的 compile_metrics bug）
            self._metric_names = ['loss', 'auc', 'precision', 'recall', 'accuracy']
        else:
            # 回归任务
            self.model.compile(
                optimizer=optimizer,
                loss='mse',
                metrics=['mae']
            )
            self._metric_names = ['loss', 'mae']

        self.is_compiled = True

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        training_config: Optional[TrainingConfig] = None,
        callbacks: Optional[list] = None,
        verbose: int = 1
    ) -> Dict:
        """
        训练模型

        参数:
            X_train, y_train: 训练数据
            X_val, y_val: 验证数据（可选）
            training_config: 训练配置
            callbacks: 自定义回调列表（可选）
            verbose: 显示详细程度

        返回:
            训练历史记录
        """
        if training_config is None:
            training_config = TrainingConfig()

        if not self.is_compiled:
            self.compile()

        if callbacks is None:
            callbacks = self._create_callbacks(training_config)

        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)

        self.history = self.model.fit(
            X_train,
            y_train,
            batch_size=training_config.batch_size,
            epochs=training_config.epochs,
            validation_data=validation_data,
            validation_split=training_config.validation_split if validation_data is None else 0.0,
            callbacks=callbacks,
            verbose=verbose
        )

        return self.history.history

    def _create_callbacks(self, training_config: TrainingConfig):
        """创建训练回调"""
        callbacks = []

        # 早停 - 监控AUC
        if training_config.early_stopping_patience > 0:
            monitor_metric = 'val_auc' if self.task_type == 'classification' else 'val_loss'
            mode = 'max' if self.task_type == 'classification' else 'min'
            callbacks.append(EarlyStopping(
                monitor=monitor_metric,
                mode=mode,
                patience=training_config.early_stopping_patience,
                restore_best_weights=True,
                verbose=1
            ))

        # 学习率衰减
        if training_config.reduce_lr_patience > 0:
            monitor_metric = 'val_auc' if self.task_type == 'classification' else 'val_loss'
            mode = 'max' if self.task_type == 'classification' else 'min'
            callbacks.append(ReduceLROnPlateau(
                monitor=monitor_metric,
                mode=mode,
                factor=training_config.reduce_lr_factor,
                patience=training_config.reduce_lr_patience,
                min_lr=training_config.min_learning_rate,
                verbose=1
            ))

        # 模型检查点
        if training_config.save_dir:
            save_dir = Path(training_config.save_dir) / self.name
            save_dir.mkdir(parents=True, exist_ok=True)
            monitor_metric = 'val_auc' if self.task_type == 'classification' else 'val_loss'
            mode = 'max' if self.task_type == 'classification' else 'min'
            callbacks.append(ModelCheckpoint(
                filepath=str(save_dir / "best_model.h5"),
                monitor=monitor_metric,
                mode=mode,
                save_best_only=training_config.save_best_only,
                verbose=1
            ))

        return callbacks

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """评估模型"""
        if self.model is None:
            raise ValueError("Model not built")
        results = self.model.evaluate(X_test, y_test, verbose=0)

        # model.metrics_names可能返回['loss', 'compile_metrics']（TF bug），
        # 使用编译时存储的指标名列表
        if hasattr(self, '_metric_names') and len(self._metric_names) == len(results):
            metrics = dict(zip(self._metric_names, results))
        else:
            metrics = {}
            for i, name in enumerate(self.model.metrics_names):
                if i < len(results):
                    metrics[name] = results[i]
        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """预测"""
        if self.model is None:
            raise ValueError("Model not built")
        return self.model.predict(X, verbose=0)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """预测概率（仅分类任务）"""
        if self.task_type != "classification":
            raise ValueError("predict_proba only available for classification tasks")
        return self.predict(X)

    def save(self, filepath: str):
        """保存模型"""
        if self.model is None:
            raise ValueError("Model not built")
        save_dir = Path(filepath).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        self.model.save(filepath)

    def load_weights(self, filepath: str):
        """加载权重"""
        if self.model is None:
            self.build_model()
        self.model.load_weights(filepath)

    def summary(self):
        """打印模型摘要"""
        if self.model:
            self.model.summary()
