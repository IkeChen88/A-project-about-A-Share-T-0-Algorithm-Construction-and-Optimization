"""
模型配置文件
包含所有模型的超参数配置
"""
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ModelConfig:
    """基础模型配置"""
    input_shape: tuple = (20, 33)
    output_dim: int = 1
    dropout_rate: float = 0.3
    l2_reg: float = 1e-4
    learning_rate: float = 1e-3
    batch_size: int = 64
    epochs: int = 100
    early_stopping_patience: int = 10
    reduce_lr_patience: int = 5


@dataclass
class LSTMConfig(ModelConfig):
    """LSTM模型配置"""
    lstm_units: List[int] = (64, 32)
    bidirectional: bool = False
    use_residual: bool = True
    use_batch_norm: bool = True


@dataclass
class TransformerLSTMConfig(ModelConfig):
    """Transformer-LSTM混合模型配置"""
    d_model: int = 64
    num_heads: int = 4
    ff_dim: int = 128
    num_transformer_blocks: int = 2
    dropout_rate: float = 0.2
    lstm_units: List[int] = (32,)
    use_residual: bool = True
    use_batch_norm: bool = True


@dataclass
class CNNConfig(ModelConfig):
    """CNN模型配置"""
    filters_list: List[int] = (64, 32)
    kernel_sizes: List[int] = (3, 3)
    pool_sizes: List[int] = (2, 1)
    use_residual: bool = True
    use_batch_norm: bool = True


@dataclass
class MLPConfig(ModelConfig):
    """MLP模型配置"""
    hidden_units: List[int] = (128, 64, 32)
    use_residual: bool = True
    use_batch_norm: bool = True


@dataclass
class EnsembleConfig:
    """集成模型配置"""
    use_weighted_average: bool = True
    use_stacking: bool = False
    stacking_units: int = 16
    weights: Optional[List[float]] = None


@dataclass
class TrainingConfig:
    """训练配置"""
    batch_size: int = 64
    epochs: int = 100
    validation_split: float = 0.2
    early_stopping_patience: int = 15
    reduce_lr_patience: int = 7
    reduce_lr_factor: float = 0.5
    min_learning_rate: float = 1e-6
    save_best_only: bool = True
    save_dir: str = "saved_models"
    log_dir: str = "logs"


DEFAULT_LSTM_CONFIG = LSTMConfig()
DEFAULT_TRANSFORMER_LSTM_CONFIG = TransformerLSTMConfig()
DEFAULT_CNN_CONFIG = CNNConfig()
DEFAULT_MLP_CONFIG = MLPConfig()
DEFAULT_ENSEMBLE_CONFIG = EnsembleConfig()
DEFAULT_TRAINING_CONFIG = TrainingConfig()
