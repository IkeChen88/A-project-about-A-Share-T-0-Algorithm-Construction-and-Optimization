"""
超参数优化模块
使用Optuna进行超参数搜索
支持时间序列交叉验证
支持多种模型类型: LSTM, Transformer-LSTM, CNN, MLP
"""
import numpy as np
import tensorflow as tf
import optuna
from optuna.trial import Trial
from typing import Dict, Callable, Optional, Tuple, Union, List
from pathlib import Path
import json
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from models.lstm_model import LSTMModel
from models.transformer_lstm import TransformerLSTMModel
from models.cnn_model import CNNModel
from models.mlp_model import MLPModel
from models.ts_cross_validation import TimeSeriesCrossValidator
from config.model_config import (
    LSTMConfig, TransformerLSTMConfig, CNNConfig, MLPConfig, TrainingConfig
)


class OptunaOptimizer:
    """
    Optuna超参数优化器
    """

    # 模型类型到采样方法的分发映射
    _SAMPLER_MAP = {
        "lstm": "_sample_lstm_params",
        "transformer_lstm": "_sample_transformer_lstm_params",
        "cnn": "_sample_cnn_params",
        "mlp": "_sample_mlp_params",
    }

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        input_shape: Tuple[int, ...] = None,
        task_type: str = "classification",
        direction: str = "maximize",
        study_name: str = "lstm_optimization",
        storage: Optional[str] = None,
        model_type: str = "lstm",
    ):
        """
        初始化优化器

        参数:
            X_train, y_train: 训练数据
            X_val, y_val: 验证数据（可选，用于最终评估）
            input_shape: 输入形状
            task_type: 任务类型 ("classification" 或 "regression")
            direction: 优化方向 ("maximize" 或 "minimize")
            study_name: 研究名称
            storage: 存储路径（可选，用于持久化）
            model_type: 模型类型 ("lstm", "transformer_lstm", "cnn", "mlp")
        """
        if model_type not in self._SAMPLER_MAP:
            raise ValueError(f"Unknown model_type: {model_type}. Must be one of {list(self._SAMPLER_MAP.keys())}")

        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.input_shape = input_shape or X_train.shape[1:]
        self.task_type = task_type
        self.direction = direction
        self.study_name = study_name
        self.storage = storage
        self.model_type = model_type

        self.study = None
        self.best_params = None
        self.best_value = None

    def objective(self, trial: Trial) -> float:
        """
        Optuna目标函数

        参数:
            trial: Optuna Trial对象

        返回:
            要优化的指标值
        """
        # 采样超参数
        params = self._sample_params(trial)

        # 打印当前试验参数
        logger.info(f"\n{'='*60}")
        logger.info(f"Trial {trial.number}")
        logger.info(f"Parameters: {params}")
        logger.info(f"{'='*60}")

        # 构建模型配置和模型类
        config, ModelClass = self._build_config_and_class(params)

        # 创建模型构建函数
        def model_builder():
            model = ModelClass(config, task_type=self.task_type, name=f"{self.model_type}_trial_{trial.number}")
            model.compile(learning_rate=params['learning_rate'])
            return model

        # 训练配置
        training_config = TrainingConfig(
            batch_size=params['batch_size'],
            epochs=100,
            early_stopping_patience=15,
            reduce_lr_patience=7,
            reduce_lr_factor=0.5,
            min_learning_rate=1e-7,
            save_dir=None  # 不保存中间模型
        )

        # 使用时间序列交叉验证
        use_cv = params.get('use_cv', True)

        if use_cv and len(self.X_train) > 1000:
            # 时间序列交叉验证
            cv = TimeSeriesCrossValidator(
                n_splits=3,
                val_window_size=min(200, len(self.X_train) // 10),
                fixed_size=True,
                task_type=self.task_type
            )
            results = cv.cross_validate(
                model_builder,
                self.X_train,
                self.y_train,
                X_val_fixed=self.X_val,
                y_val_fixed=self.y_val,
                training_config=training_config,
                verbose=0
            )
            # 获取平均AUC — 灵活查找key
            if self.task_type == 'classification':
                # 查找以'auc'开头的summary key
                auc_keys = [k for k in results['summary'].keys() if k.startswith('auc_mean')]
                score = results['summary'][auc_keys[0]] if auc_keys else results['summary'].get('auc_mean', 0.5)
            else:
                score = -results['summary']['loss_mean']  # 最小化loss
        else:
            # 单次训练
            model = model_builder()
            model.train(
                self.X_train, self.y_train,
                self.X_val, self.y_val,
                training_config=training_config,
                verbose=1
            )
            metrics = model.evaluate(self.X_val, self.y_val) if self.X_val is not None else model.evaluate(self.X_train, self.y_train)

            if self.task_type == 'classification':
                score = metrics['auc']
            else:
                score = -metrics['loss']

        logger.info(f"Trial {trial.number} score: {score:.6f}")

        return score

    # ==================== 采样方法 ====================

    def _sample_params(self, trial: Trial) -> Dict:
        """
        从搜索空间采样超参数（根据model_type分发）

        参数:
            trial: Optuna Trial对象

        返回:
            参数字典
        """
        method_name = self._SAMPLER_MAP[self.model_type]
        return getattr(self, method_name)(trial)

    def _sample_shared_params(self, trial: Trial) -> Dict:
        """采样所有模型共享的超参数"""
        return {
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5, step=0.1),
            'l2_reg': trial.suggest_float('l2_reg', 1e-5, 1e-3, log=True),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128]),
            'use_residual': trial.suggest_categorical('use_residual', [True, False]),
            'use_batch_norm': trial.suggest_categorical('use_batch_norm', [True, False]),
        }

    def _sample_lstm_params(self, trial: Trial) -> Dict:
        """
        采样LSTM模型超参数

        参数:
            trial: Optuna Trial对象

        返回:
            参数字典
        """
        params = self._sample_shared_params(trial)

        # LSTM结构参数
        n_layers = trial.suggest_int('n_layers', 1, 3)
        lstm_units = []
        for i in range(n_layers):
            units = trial.suggest_categorical(f'lstm_units_{i}', [32, 64, 128, 256])
            lstm_units.append(units)

        # LSTM特有参数
        bidirectional = trial.suggest_categorical('bidirectional', [True, False])

        params.update({
            'n_layers': n_layers,
            'lstm_units': lstm_units,
            'bidirectional': bidirectional,
        })
        return params

    def _sample_transformer_lstm_params(self, trial: Trial) -> Dict:
        """
        采样Transformer-LSTM混合模型超参数

        参数:
            trial: Optuna Trial对象

        返回:
            参数字典
        """
        params = self._sample_shared_params(trial)

        # Transformer特定参数
        d_model = trial.suggest_categorical('d_model', [32, 64, 128])
        # num_heads必须整除d_model
        valid_heads = [h for h in [2, 4, 8] if d_model % h == 0]
        num_heads = trial.suggest_categorical('num_heads', valid_heads)
        ff_dim = trial.suggest_categorical('ff_dim', [64, 128, 256])
        num_transformer_blocks = trial.suggest_int('num_transformer_blocks', 1, 3)

        # LSTM层参数
        n_lstm_layers = trial.suggest_int('n_lstm_layers', 1, 3)
        lstm_units = []
        for i in range(n_lstm_layers):
            units = trial.suggest_categorical(f'lstm_units_{i}', [32, 64, 128])
            lstm_units.append(units)

        params.update({
            'd_model': d_model,
            'num_heads': num_heads,
            'ff_dim': ff_dim,
            'num_transformer_blocks': num_transformer_blocks,
            'n_lstm_layers': n_lstm_layers,
            'lstm_units': lstm_units,
        })
        return params

    def _sample_cnn_params(self, trial: Trial) -> Dict:
        """
        采样CNN模型超参数

        参数:
            trial: Optuna Trial对象

        返回:
            参数字典
        """
        params = self._sample_shared_params(trial)

        # CNN结构参数
        n_conv_layers = trial.suggest_int('n_conv_layers', 1, 3)
        filters_list = []
        kernel_sizes = []
        pool_sizes = []
        for i in range(n_conv_layers):
            filters = trial.suggest_categorical(f'filters_{i}', [32, 64, 128])
            kernel_size = trial.suggest_categorical(f'kernel_size_{i}', [2, 3, 5])
            pool_size = trial.suggest_categorical(f'pool_size_{i}', [1, 2])
            filters_list.append(filters)
            kernel_sizes.append(kernel_size)
            pool_sizes.append(pool_size)

        params.update({
            'n_conv_layers': n_conv_layers,
            'filters_list': filters_list,
            'kernel_sizes': kernel_sizes,
            'pool_sizes': pool_sizes,
        })
        return params

    def _sample_mlp_params(self, trial: Trial) -> Dict:
        """
        采样MLP模型超参数

        参数:
            trial: Optuna Trial对象

        返回:
            参数字典
        """
        params = self._sample_shared_params(trial)

        # MLP结构参数
        n_hidden_layers = trial.suggest_int('n_hidden_layers', 1, 4)
        hidden_units = []
        for i in range(n_hidden_layers):
            units = trial.suggest_categorical(f'hidden_units_{i}', [32, 64, 128, 256])
            hidden_units.append(units)

        params.update({
            'n_hidden_layers': n_hidden_layers,
            'hidden_units': hidden_units,
        })
        return params

    # ==================== 配置构建 ====================

    def _build_config_and_class(self, params: Dict) -> Tuple:
        """
        根据model_type和采样参数构建对应的模型配置和模型类

        参数:
            params: 采样参数字典

        返回:
            (config, ModelClass) 元组
        """
        if self.model_type == "lstm":
            config = LSTMConfig(
                input_shape=self.input_shape,
                output_dim=1,
                lstm_units=params['lstm_units'],
                dropout_rate=params['dropout_rate'],
                l2_reg=params['l2_reg'],
                learning_rate=params['learning_rate'],
                bidirectional=params.get('bidirectional', False),
                use_residual=params['use_residual'],
                use_batch_norm=params['use_batch_norm'],
            )
            ModelClass = LSTMModel
        elif self.model_type == "transformer_lstm":
            config = TransformerLSTMConfig(
                input_shape=self.input_shape,
                output_dim=1,
                d_model=params['d_model'],
                num_heads=params['num_heads'],
                ff_dim=params['ff_dim'],
                num_transformer_blocks=params['num_transformer_blocks'],
                lstm_units=params['lstm_units'],
                dropout_rate=params['dropout_rate'],
                l2_reg=params['l2_reg'],
                learning_rate=params['learning_rate'],
                use_residual=params['use_residual'],
                use_batch_norm=params['use_batch_norm'],
            )
            ModelClass = TransformerLSTMModel
        elif self.model_type == "cnn":
            config = CNNConfig(
                input_shape=self.input_shape,
                output_dim=1,
                filters_list=params['filters_list'],
                kernel_sizes=params['kernel_sizes'],
                pool_sizes=params['pool_sizes'],
                dropout_rate=params['dropout_rate'],
                l2_reg=params['l2_reg'],
                learning_rate=params['learning_rate'],
                use_residual=params['use_residual'],
                use_batch_norm=params['use_batch_norm'],
            )
            ModelClass = CNNModel
        elif self.model_type == "mlp":
            config = MLPConfig(
                input_shape=self.input_shape,
                output_dim=1,
                hidden_units=params['hidden_units'],
                dropout_rate=params['dropout_rate'],
                l2_reg=params['l2_reg'],
                learning_rate=params['learning_rate'],
                use_residual=params['use_residual'],
                use_batch_norm=params['use_batch_norm'],
            )
            ModelClass = MLPModel
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        return config, ModelClass

    def optimize(
        self,
        n_trials: int = 50,
        timeout: Optional[int] = None,
        n_jobs: int = 1,
        show_progress_bar: bool = True
    ) -> Dict:
        """
        执行超参数优化

        参数:
            n_trials: 试验次数
            timeout: 超时时间（秒）
            n_jobs: 并行任务数
            show_progress_bar: 是否显示进度条

        返回:
            最佳参数字典
        """
        logger.info(f"开始超参数优化: n_trials={n_trials}")

        # 创建或加载Study
        if self.study is None:
            self.study = optuna.create_study(
                study_name=self.study_name,
                direction=self.direction,
                storage=self.storage,
                load_if_exists=True,
                sampler=optuna.samplers.TPESampler(seed=42)
            )

        # 运行优化
        self.study.optimize(
            self.objective,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
            show_progress_bar=show_progress_bar,
            gc_after_trial=True
        )

        # 保存结果
        self.best_params = self.study.best_params
        self.best_value = self.study.best_value

        logger.info(f"\n{'='*60}")
        logger.info(f"优化完成!")
        logger.info(f"最佳值: {self.best_value:.6f}")
        logger.info(f"最佳参数: {self.best_params}")
        logger.info(f"{'='*60}")

        return self.best_params

    def _reconstruct_list_params(self, params: Dict) -> Dict:
        """
        从Optuna存储的独立key（如lstm_units_0, lstm_units_1）重建列表参数

        参数:
            params: Optuna最佳参数字典

        返回:
            重建后的参数字典
        """
        params = dict(params)  # 复制

        if self.model_type == "lstm":
            n_layers = params.get('n_layers', 1)
            params['lstm_units'] = [params[f'lstm_units_{i}'] for i in range(n_layers)]
        elif self.model_type == "transformer_lstm":
            n_lstm_layers = params.get('n_lstm_layers', 1)
            params['lstm_units'] = [params[f'lstm_units_{i}'] for i in range(n_lstm_layers)]
        elif self.model_type == "cnn":
            n_conv_layers = params.get('n_conv_layers', 1)
            params['filters_list'] = [params[f'filters_{i}'] for i in range(n_conv_layers)]
            params['kernel_sizes'] = [params[f'kernel_size_{i}'] for i in range(n_conv_layers)]
            params['pool_sizes'] = [params[f'pool_size_{i}'] for i in range(n_conv_layers)]
        elif self.model_type == "mlp":
            n_hidden_layers = params.get('n_hidden_layers', 1)
            params['hidden_units'] = [params[f'hidden_units_{i}'] for i in range(n_hidden_layers)]

        return params

    def get_best_model_config(self) -> Union[LSTMConfig, TransformerLSTMConfig, CNNConfig, MLPConfig]:
        """
        从最佳参数构建模型配置

        返回:
            对应model_type的配置实例
        """
        if self.best_params is None:
            raise ValueError("No optimization has been run yet")

        # 重建列表参数
        reconstructed = self._reconstruct_list_params(self.best_params)
        config, _ = self._build_config_and_class(reconstructed)
        return config

    def get_training_config(self) -> TrainingConfig:
        """
        从最佳参数构建TrainingConfig

        返回:
            TrainingConfig实例
        """
        if self.best_params is None:
            raise ValueError("No optimization has been run yet")

        return TrainingConfig(
            batch_size=self.best_params['batch_size'],
            epochs=100,
            early_stopping_patience=15,
            reduce_lr_patience=7,
            reduce_lr_factor=0.5,
            min_learning_rate=1e-7,
            save_dir="saved_models"
        )

    def save_study(self, filepath: str):
        """保存Study结果"""
        if self.study is None:
            raise ValueError("No study exists")

        save_path = Path(filepath)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # 保存Study信息
        study_info = {
            'model_type': self.model_type,
            'best_params': self.best_params,
            'best_value': self.best_value,
            'n_trials': len(self.study.trials),
            'trials': []
        }

        for trial in self.study.trials:
            trial_info = {
                'number': trial.number,
                'value': trial.value,
                'params': trial.params,
                'state': str(trial.state),
                'datetime_start': str(trial.datetime_start),
                'datetime_complete': str(trial.datetime_complete)
            }
            study_info['trials'].append(trial_info)

        with open(save_path, 'w') as f:
            json.dump(study_info, f, indent=2, default=str)

        # 也保存为CSV格式的参数重要性
        try:
            importance = optuna.importance.get_param_importances(self.study)
            importance_df = {
                'param': list(importance.keys()),
                'importance': list(importance.values())
            }
            import pandas as pd
            pd.DataFrame(importance_df).to_csv(
                save_path.parent / f"{save_path.stem}_importance.csv",
                index=False
            )
        except Exception as e:
            logger.warning(f"Could not save parameter importance: {e}")

    def plot_optimization_history(self, filepath: str = None):
        """绘制优化历史"""
        try:
            import plotly
            fig = optuna.visualization.plot_optimization_history(self.study)
            if filepath:
                fig.write_html(filepath)
            return fig
        except Exception as e:
            logger.warning(f"Could not plot optimization history: {e}")
            return None

    def plot_param_importances(self, filepath: str = None):
        """绘制参数重要性"""
        try:
            import plotly
            fig = optuna.visualization.plot_param_importances(self.study)
            if filepath:
                fig.write_html(filepath)
            return fig
        except Exception as e:
            logger.warning(f"Could not plot parameter importances: {e}")
            return None

    def plot_parallel_coordinate(self, filepath: str = None):
        """绘制平行坐标图"""
        try:
            import plotly
            fig = optuna.visualization.plot_parallel_coordinate(self.study)
            if filepath:
                fig.write_html(filepath)
            return fig
        except Exception as e:
            logger.warning(f"Could not plot parallel coordinate: {e}")
            return None


def create_optimizer_from_data(
    data_dir: str = None,
    task_type: str = "classification",
    study_name: str = "lstm_optimization",
    model_type: str = "lstm",
) -> OptunaOptimizer:
    """
    从数据创建优化器的便捷函数

    参数:
        data_dir: 数据目录
        task_type: 任务类型
        study_name: 研究名称
        model_type: 模型类型 ("lstm", "transformer_lstm", "cnn", "mlp")

    返回:
        OptunaOptimizer实例
    """
    from data.data_helper import load_processed_data, normalize_data

    # 加载数据
    data = load_processed_data(data_dir)

    # 标准化数据
    X_train, X_val, X_test = normalize_data(
        data['X_train'], data['X_val'], data['X_test']
    )

    return OptunaOptimizer(
        X_train=X_train,
        y_train=data['y_train'],
        X_val=X_val,
        y_val=data['y_val'],
        input_shape=data['input_shape'],
        task_type=task_type,
        direction="maximize" if task_type == "classification" else "minimize",
        study_name=study_name,
        model_type=model_type,
    )
