"""
基础模型类
包含通用训练、评估、保存/加载功能
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
import os
from pathlib import Path
from typing import Dict, Optional
import json

from config.model_config import ModelConfig, TrainingConfig


class BaseModel:
    """基础模型类"""

    def __init__(self, config: ModelConfig, name: str = "base_model"):
        self.config = config
        self.name = name
        self.model: Optional[Model] = None
        self.history = None
        self.is_compiled = False

    def build_model(self):
        raise NotImplementedError("Subclasses must implement build_model")

    def compile(self, optimizer: Optional[tf.keras.optimizers.Optimizer] = None):
        if self.model is None:
            self.build_model()

        if optimizer is None:
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.config.learning_rate)

        self.model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
        self.is_compiled = True

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        training_config: Optional[TrainingConfig] = None
    ) -> Dict:
        if training_config is None:
            training_config = TrainingConfig()

        if not self.is_compiled:
            self.compile()

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
            verbose=1
        )

        return self.history.history

    def _create_callbacks(self, training_config: TrainingConfig):
        callbacks = []

        if training_config.early_stopping_patience > 0:
            callbacks.append(EarlyStopping(
                monitor="val_loss",
                patience=training_config.early_stopping_patience,
                restore_best_weights=True
            ))

        if training_config.reduce_lr_patience > 0:
            callbacks.append(ReduceLROnPlateau(
                monitor="val_loss",
                factor=training_config.reduce_lr_factor,
                patience=training_config.reduce_lr_patience,
                min_lr=training_config.min_learning_rate
            ))

        if training_config.save_dir:
            save_dir = Path(training_config.save_dir) / self.name
            save_dir.mkdir(parents=True, exist_ok=True)
            callbacks.append(ModelCheckpoint(
                filepath=str(save_dir / "best_model.h5"),
                monitor="val_loss",
                save_best_only=training_config.save_best_only
            ))

        return callbacks

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        if self.model is None:
            raise ValueError("Model not built")
        results = self.model.evaluate(X_test, y_test, verbose=0)
        metrics = {}
        for i, name in enumerate(self.model.metrics_names):
            metrics[name] = results[i]
        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not built")
        return self.model.predict(X, verbose=0)

    def save(self, filepath: str):
        if self.model is None:
            raise ValueError("Model not built")
        save_dir = Path(filepath).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        self.model.save(filepath)

    def summary(self):
        if self.model:
            self.model.summary()
