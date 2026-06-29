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

    def optimize_and_train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        model_types: Optional[List[str]] = None,
        n_trials: int = 30,
        input_shape: Optional[tuple] = None,
    ) -> Dict[str, Dict]:
        """
        对每种模型类型执行Optuna超参数优化，用最佳参数训练，自动创建集成模型

        参数:
            X_train, y_train: 训练数据
            X_val, y_val: 验证数据
            X_test, y_test: 测试数据（可选）
            model_types: 要优化的模型类型列表，默认所有
            n_trials: 每种模型的Optuna试验次数
            input_shape: 输入形状，默认从X_train推断

        返回:
            Dict[str, Dict]: 每个模型的优化结果
        """
        from models.hyperparameter_optimizer import OptunaOptimizer
        from models.lstm_model import LSTMModel
        from models.transformer_lstm import TransformerLSTMModel
        from models.cnn_model import CNNModel
        from models.mlp_model import MLPModel

        if model_types is None:
            model_types = ["lstm", "transformer_lstm", "cnn", "mlp"]

        if input_shape is None:
            input_shape = X_train.shape[1:]

        # 模型类映射
        MODEL_CLASS_MAP = {
            "lstm": LSTMModel,
            "transformer_lstm": TransformerLSTMModel,
            "cnn": CNNModel,
            "mlp": MLPModel,
        }

        optimization_results = {}

        for mt in model_types:
            print(f"\n{'='*70}")
            print(f"优化 {mt} 模型...")
            print('='*70)

            # 创建优化器
            optimizer = OptunaOptimizer(
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                input_shape=input_shape,
                task_type=self.task_type,
                direction="maximize" if self.task_type == "classification" else "minimize",
                study_name=f"{mt}_optimization",
                model_type=mt,
            )

            # 执行超参数优化
            best_params = optimizer.optimize(
                n_trials=n_trials,
                show_progress_bar=True,
            )

            # 获取最佳配置
            best_config = optimizer.get_best_model_config()
            best_training_config = optimizer.get_training_config()
            best_training_config.save_dir = str(Path("saved_models") / mt)

            # 用最佳参数训练模型
            print(f"\n用最佳参数训练 {mt} 模型...")
            ModelClass = MODEL_CLASS_MAP[mt]
            model = ModelClass(best_config, task_type=self.task_type, name=f"{mt}_optimized")
            model.compile(learning_rate=best_config.learning_rate)
            model.summary()

            history = model.train(
                X_train, y_train,
                X_val, y_val,
                training_config=best_training_config,
                verbose=1,
            )

            # 评估
            val_metrics = model.evaluate(X_val, y_val)
            test_metrics = None
            if X_test is not None and y_test is not None:
                test_metrics = model.evaluate(X_test, y_test)

            # 存储结果
            result = {
                "config": best_config,
                "training_config": best_training_config,
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
                "history": history,
                "optimizer": optimizer,
            }
            optimization_results[mt] = result
            self.models[mt] = model
            self.histories[mt] = history
            self.results[mt] = val_metrics

            print(f"\n{mt} 验证集结果:")
            for k, v in val_metrics.items():
                print(f"  {k}: {v:.6f}")

        # 选出最佳模型
        if self.task_type == "classification":
            best_model_name = max(optimization_results, key=lambda k: optimization_results[k]["val_metrics"]["auc"])
        else:
            best_model_name = min(optimization_results, key=lambda k: optimization_results[k]["val_metrics"]["loss"])

        self.best_model_name = best_model_name
        self.best_model_score = optimization_results[best_model_name]["val_metrics"]

        print(f"\n{'='*70}")
        print(f"最佳模型: {best_model_name}")
        print(f"验证集指标: {self.best_model_score}")
        print('='*70)

        # 创建集成模型
        print(f"\n创建集成模型...")
        ensemble = self.create_ensemble(list(optimization_results.keys()))
        ensemble.compile(learning_rate=1e-4)
        if X_test is not None and y_test is not None:
            ensemble_metrics = ensemble.evaluate(X_test, y_test)
            print(f"\n集成模型测试集结果:")
            for k, v in ensemble_metrics.items():
                print(f"  {k}: {v:.6f}")
            optimization_results["ensemble"] = {
                "test_metrics": ensemble_metrics,
            }

        return optimization_results

    def compare_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        model_types: Optional[List[str]] = None,
        n_trials: int = 30,
    ) -> pd.DataFrame:
        """
        对比多种模型 — 优化、训练、评估并打印对比表

        参数:
            X_train, y_train: 训练数据
            X_val, y_val: 验证数据
            X_test, y_test: 测试数据（可选）
            model_types: 模型类型列表
            n_trials: Optuna试验次数

        返回:
            对比结果DataFrame
        """
        # 执行优化和训练
        results = self.optimize_and_train(
            X_train, y_train, X_val, y_val,
            X_test=X_test, y_test=y_test,
            model_types=model_types,
            n_trials=n_trials,
        )

        # 构建对比表
        rows = []
        for name, result in results.items():
            if name == "ensemble":
                row = {"Model": name}
                if result.get("test_metrics"):
                    for k, v in result["test_metrics"].items():
                        row[k] = round(v, 4)
            else:
                row = {"Model": name}
                metrics = result.get("test_metrics") or result["val_metrics"]
                for k, v in metrics.items():
                    row[k] = round(v, 4)
            rows.append(row)

        comparison_df = pd.DataFrame(rows)
        comparison_df = comparison_df.set_index("Model")

        print(f"\n{'='*70}")
        print("模型对比结果")
        print('='*70)
        print(comparison_df.to_string())
        print(f"\n最佳模型: {self.best_model_name}")

        return comparison_df

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
