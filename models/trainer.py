"""
训练器类
用于统一训练和评估所有模型
"""
import numpy as np
import tensorflow as tf
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import json

from models.base_model import BaseModel
from models.lstm_model import LSTMModel
from models.transformer_lstm import TransformerLSTMModel
from models.cnn_model import CNNModel
from models.mlp_model import MLPModel
from models.ensemble_model import EnsembleModel
from config.model_config import (
    DEFAULT_LSTM_CONFIG,
    DEFAULT_TRANSFORMER_LSTM_CONFIG,
    DEFAULT_CNN_CONFIG,
    DEFAULT_MLP_CONFIG,
    TrainingConfig,
    DEFAULT_TRAINING_CONFIG
)


class ModelTrainer:
    """模型训练器"""

    def __init__(
        self,
        training_config: TrainingConfig = DEFAULT_TRAINING_CONFIG,
        task_type: str = "classification"
    ):
        self.training_config = training_config
        self.task_type = task_type
        self.models: Dict[str, BaseModel] = {}
        self.histories: Dict[str, Dict] = {}
        self.results: Dict[str, Dict] = {}

    def create_all_models(self, input_shape: tuple = (20, 33)):
        lstm_config = DEFAULT_LSTM_CONFIG
        lstm_config.input_shape = input_shape
        self.models["lstm"] = LSTMModel(lstm_config, task_type=self.task_type)

        transformer_config = DEFAULT_TRANSFORMER_LSTM_CONFIG
        transformer_config.input_shape = input_shape
        self.models["transformer_lstm"] = TransformerLSTMModel(transformer_config, task_type=self.task_type)

        cnn_config = DEFAULT_CNN_CONFIG
        cnn_config.input_shape = input_shape
        self.models["cnn"] = CNNModel(cnn_config, task_type=self.task_type)

        mlp_config = DEFAULT_MLP_CONFIG
        mlp_config.input_shape = input_shape
        self.models["mlp"] = MLPModel(mlp_config, task_type=self.task_type)

    def train_all(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ):
        for name, model in self.models.items():
            print(f"\n{'='*60}")
            print(f"Training {name} model...")
            print('='*60)

            model.build_model()
            model.summary()

            history = model.train(X_train, y_train, X_val, y_val, self.training_config)
            self.histories[name] = history

        print("\nAll models trained!")

    def evaluate_all(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        print(f"\n{'='*60}")
        print("Evaluating all models...")
        print('='*60)

        for name, model in self.models.items():
            metrics = model.evaluate(X_test, y_test)
            self.results[name] = metrics
            print(f"\n{name}:")
            for metric_name, value in metrics.items():
                print(f"  {metric_name}: {value:.6f}")

        return self.results

    def create_ensemble(self, model_names: Optional[List[str]] = None) -> EnsembleModel:
        if model_names is None:
            model_names = list(self.models.keys())

        base_models = [self.models[name] for name in model_names]
        ensemble = EnsembleModel(base_models, task_type=self.task_type)
        ensemble.build_model()

        self.models["ensemble"] = ensemble
        return ensemble

    def save_results(self, save_dir: str = "results"):
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        if self.results:
            results_df = pd.DataFrame(self.results).T
            results_df.to_csv(save_path / "evaluation_results.csv")

        for name, model in self.models.items():
            model_path = save_path / f"{name}_model.h5"
            model.save(str(model_path))

        print(f"Results saved to {save_path}")

    def summary(self):
        for name, model in self.models.items():
            print(f"\n{'='*60}")
            print(f"Model: {name}")
            print('='*60)
            model.summary()
