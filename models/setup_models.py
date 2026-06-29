"""
自动化模型模块安装脚本
运行此脚本将自动创建所有模型相关文件
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

FILES = {}

# ==========================================
# 1. requirements.txt
# ==========================================
FILES["requirements.txt"] = """# VeighNa量化平台
vnpy>=3.0.0

# 数据处理
pandas>=2.0.0
numpy>=1.24.0

# 技术指标
ta-lib>=0.4.28
pandas-ta>=0.3.14

# 机器学习和特征工程
scikit-learn>=1.3.0
tsfresh>=0.20.0

# 深度学习框架
tensorflow>=2.13.0
keras>=2.13.0

# 可视化
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.14.0

# Jupyter
jupyter>=1.0.0
jupyterlab>=3.6.0

# 其他工具
tqdm>=4.65.0
pyarrow>=12.0.0
scipy>=1.10.0
"""

# ==========================================
# 2. config/model_config.py
# ==========================================
FILES["config/model_config.py"] = '''"""
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
'''

# ==========================================
# 3. models/__init__.py
# ==========================================
FILES["models/__init__.py"] = '''"""
模型模块
包含所有深度学习模型和集成系统
"""
from models.base_model import BaseModel
from models.lstm_model import LSTMModel
from models.transformer_lstm import TransformerLSTMModel
from models.cnn_model import CNNModel
from models.mlp_model import MLPModel
from models.ensemble_model import EnsembleModel
from models.trainer import ModelTrainer

__all__ = [
    "BaseModel",
    "LSTMModel",
    "TransformerLSTMModel",
    "CNNModel",
    "MLPModel",
    "EnsembleModel",
    "ModelTrainer"
]
'''

# ==========================================
# 4. models/initializers.py
# ==========================================
FILES["models/initializers.py"] = '''"""
自定义权重初始化器
包含Xavier (Glorot) 初始化和正交初始化
"""
import tensorflow as tf
from tensorflow.keras import initializers
from tensorflow.keras.initializers import Initializer


class XavierInitializer(Initializer):
    """Xavier (Glorot) 初始化器"""

    def __init__(self, seed=None):
        self.seed = seed

    def __call__(self, shape, dtype=None, **kwargs):
        return initializers.GlorotUniform(seed=self.seed)(shape, dtype)

    def get_config(self):
        return {"seed": self.seed}


class OrthogonalInitializer(Initializer):
    """正交初始化器（特别适合LSTM循环核）"""

    def __init__(self, gain=1.0, seed=None):
        self.gain = gain
        self.seed = seed

    def __call__(self, shape, dtype=None, **kwargs):
        return initializers.Orthogonal(gain=self.gain, seed=self.seed)(shape, dtype)

    def get_config(self):
        return {"gain": self.gain, "seed": self.seed}


class HeInitializer(Initializer):
    """He初始化器（适合ReLU激活）"""

    def __init__(self, seed=None):
        self.seed = seed

    def __call__(self, shape, dtype=None, **kwargs):
        return initializers.HeUniform(seed=self.seed)(shape, dtype)

    def get_config(self):
        return {"seed": self.seed}


def get_initializer(type_name: str = "xavier", seed=None):
    """获取初始化器工厂函数"""
    if type_name == "xavier":
        return XavierInitializer(seed=seed)
    elif type_name == "orthogonal":
        return OrthogonalInitializer(seed=seed)
    elif type_name == "he":
        return HeInitializer(seed=seed)
    else:
        raise ValueError(f"Unknown initializer type: {type_name}")
'''

# ==========================================
# 5. models/layers.py
# ==========================================
FILES["models/layers.py"] = '''"""
自定义深度学习层
包含自注意力、位置编码、残差块等
"""
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer


class PositionalEncoding(Layer):
    """位置编码层"""

    def __init__(self, position: int, d_model: int, **kwargs):
        super().__init__(**kwargs)
        self.position = position
        self.d_model = d_model
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position: int, i: int, d_model: int) -> tf.Tensor:
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles

    def positional_encoding(self, position: int, d_model: int) -> tf.Tensor:
        angle_rads = self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model
        )

        sin = tf.math.sin(angle_rads[:, 0::2])
        cos = tf.math.cos(angle_rads[:, 1::2])

        pos_encoding = tf.concat([sin, cos], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

    def get_config(self):
        config = super().get_config()
        config.update({"position": self.position, "d_model": self.d_model})
        return config


class MultiHeadSelfAttention(Layer):
    """多头自注意力层"""

    def __init__(self, d_model: int, num_heads: int, dropout_rate: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate

        assert d_model % num_heads == 0

        self.depth = d_model // num_heads
        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)
        self.dense = layers.Dense(d_model)
        self.dropout = layers.Dropout(dropout_rate)

    def split_heads(self, x: tf.Tensor, batch_size: int) -> tf.Tensor:
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        batch_size = tf.shape(inputs)[0]

        q = self.split_heads(self.wq(inputs), batch_size)
        k = self.split_heads(self.wk(inputs), batch_size)
        v = self.split_heads(self.wv(inputs), batch_size)

        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        attention_weights = self.dropout(attention_weights, training=training)

        scaled_attention = tf.matmul(attention_weights, v)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        output = self.dense(concat_attention)
        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dropout_rate": self.dropout_rate
        })
        return config


class TransformerEncoderBlock(Layer):
    """Transformer编码器块"""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ff_dim: int,
        dropout_rate: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate

        self.att = MultiHeadSelfAttention(d_model, num_heads, dropout_rate)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(d_model)
        ])

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        attn_output = self.att(inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)

        return out2

    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout_rate": self.dropout_rate
        })
        return config


class ResidualBlock(Layer):
    """残差块（用于MLP）"""

    def __init__(
        self,
        units: int,
        activation: str = "relu",
        dropout_rate: float = 0.3,
        use_batch_norm: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.units = units
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm

        self.dense1 = layers.Dense(units)
        self.dense2 = layers.Dense(units)
        self.dropout = layers.Dropout(dropout_rate)
        self.layernorm = layers.LayerNormalization(epsilon=1e-6)
        self.batchnorm = layers.BatchNormalization() if use_batch_norm else None

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = self.dense1(inputs)
        if self.use_batch_norm:
            x = self.batchnorm(x, training=training)
        x = layers.Activation(self.activation)(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)

        if inputs.shape[-1] != self.units:
            inputs = layers.Dense(self.units, use_bias=False)(inputs)

        return self.layernorm(inputs + x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "activation": self.activation,
            "dropout_rate": self.dropout_rate,
            "use_batch_norm": self.use_batch_norm
        })
        return config


class ResidualLSTMBlock(Layer):
    """残差LSTM块"""

    def __init__(
        self,
        units: int,
        return_sequences: bool = True,
        dropout_rate: float = 0.3,
        use_batch_norm: bool = True,
        bidirectional: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.units = units
        self.return_sequences = return_sequences
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        self.bidirectional = bidirectional

        lstm_layer = layers.LSTM(
            units,
            return_sequences=return_sequences,
            recurrent_initializer="orthogonal"
        )

        if bidirectional:
            self.lstm = layers.Bidirectional(lstm_layer)
            output_units = units * 2
        else:
            self.lstm = lstm_layer
            output_units = units

        self.dropout = layers.Dropout(dropout_rate)
        self.layernorm = layers.LayerNormalization(epsilon=1e-6)
        self.batchnorm = layers.BatchNormalization() if use_batch_norm else None

        self.output_units = output_units

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = self.lstm(inputs, training=training)
        if self.use_batch_norm:
            x = self.batchnorm(x, training=training)
        x = self.dropout(x, training=training)

        if inputs.shape[-1] != self.output_units:
            inputs = layers.Dense(self.output_units, use_bias=False)(inputs)

        if not self.return_sequences and len(inputs.shape) == 3:
            inputs = inputs[:, -1, :]

        return self.layernorm(inputs + x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "return_sequences": self.return_sequences,
            "dropout_rate": self.dropout_rate,
            "use_batch_norm": self.use_batch_norm,
            "bidirectional": self.bidirectional
        })
        return config


class ResidualCNNBlock(Layer):
    """残差CNN块"""

    def __init__(
        self,
        filters: int,
        kernel_size: int,
        pool_size: int = 2,
        dropout_rate: float = 0.3,
        use_batch_norm: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm

        self.conv1 = layers.Conv1D(filters, kernel_size, padding="same")
        self.conv2 = layers.Conv1D(filters, kernel_size, padding="same")
        self.pooling = layers.MaxPooling1D(pool_size) if pool_size > 1 else None
        self.dropout = layers.Dropout(dropout_rate)
        self.layernorm = layers.LayerNormalization(epsilon=1e-6)
        self.batchnorm = layers.BatchNormalization() if use_batch_norm else None

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = self.conv1(inputs)
        if self.use_batch_norm:
            x = self.batchnorm(x, training=training)
        x = layers.Activation("relu")(x)
        x = self.conv2(x)

        if inputs.shape[-1] != self.filters:
            inputs = layers.Conv1D(self.filters, 1, padding="same", use_bias=False)(inputs)

        x = self.layernorm(inputs + x)

        if self.pooling:
            x = self.pooling(x)

        x = self.dropout(x, training=training)
        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "pool_size": self.pool_size,
            "dropout_rate": self.dropout_rate,
            "use_batch_norm": self.use_batch_norm
        })
        return config
'''

# ==========================================
# 6. models/base_model.py
# ==========================================
FILES["models/base_model.py"] = '''"""
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
'''

# ==========================================
# 7. models/lstm_model.py
# ==========================================
FILES["models/lstm_model.py"] = '''"""
LSTM模型实现
包含残差连接、批量归一化、Dropout
"""
import tensorflow as tf
from tensorflow.keras import Model, layers

from models.base_model import BaseModel
from models.layers import ResidualLSTMBlock
from models.initializers import get_initializer
from config.model_config import LSTMConfig, DEFAULT_LSTM_CONFIG


class LSTMModel(BaseModel):
    """LSTM模型"""

    def __init__(self, config: LSTMConfig = DEFAULT_LSTM_CONFIG, name: str = "lstm_model"):
        super().__init__(config, name)
        self.config = config

    def build_model(self):
        inputs = layers.Input(shape=self.config.input_shape)
        x = inputs

        xavier_initializer = get_initializer("xavier", seed=42)
        orthogonal_initializer = get_initializer("orthogonal", seed=42)

        for i, units in enumerate(self.config.lstm_units):
            return_sequences = (i < len(self.config.lstm_units) - 1)

            if self.config.use_residual:
                x = ResidualLSTMBlock(
                    units=units,
                    return_sequences=return_sequences,
                    dropout_rate=self.config.dropout_rate,
                    use_batch_norm=self.config.use_batch_norm,
                    bidirectional=self.config.bidirectional
                )(x)
            else:
                lstm_layer = layers.LSTM(
                    units,
                    return_sequences=return_sequences,
                    kernel_initializer=xavier_initializer,
                    recurrent_initializer=orthogonal_initializer
                )
                if self.config.bidirectional:
                    x = layers.Bidirectional(lstm_layer)(x)
                else:
                    x = lstm_layer(x)

                if self.config.use_batch_norm:
                    x = layers.BatchNormalization()(x)
                x = layers.Dropout(self.config.dropout_rate)(x)

        x = layers.Dense(
            32,
            activation="relu",
            kernel_initializer=xavier_initializer,
            kernel_regularizer=tf.keras.regularizers.l2(self.config.l2_reg)
        )(x)
        x = layers.Dropout(self.config.dropout_rate)(x)

        outputs = layers.Dense(
            self.config.output_dim,
            kernel_initializer=xavier_initializer
        )(x)

        self.model = Model(inputs=inputs, outputs=outputs, name=self.name)
        return self.model
'''

# ==========================================
# 8. models/transformer_lstm.py
# ==========================================
FILES["models/transformer_lstm.py"] = '''"""
混合Transformer-LSTM模型
结合自注意力机制和时序建模能力
"""
import tensorflow as tf
from tensorflow.keras import Model, layers

from models.base_model import BaseModel
from models.layers import PositionalEncoding, TransformerEncoderBlock, ResidualLSTMBlock
from models.initializers import get_initializer
from config.model_config import TransformerLSTMConfig, DEFAULT_TRANSFORMER_LSTM_CONFIG


class TransformerLSTMModel(BaseModel):
    """混合Transformer-LSTM模型"""

    def __init__(
        self,
        config: TransformerLSTMConfig = DEFAULT_TRANSFORMER_LSTM_CONFIG,
        name: str = "transformer_lstm_model"
    ):
        super().__init__(config, name)
        self.config = config

    def build_model(self):
        inputs = layers.Input(shape=self.config.input_shape)
        x = inputs

        xavier_initializer = get_initializer("xavier", seed=42)

        x = layers.Dense(self.config.d_model, kernel_initializer=xavier_initializer)(x)
        x = PositionalEncoding(position=self.config.input_shape[0], d_model=self.config.d_model)(x)

        for _ in range(self.config.num_transformer_blocks):
            x = TransformerEncoderBlock(
                d_model=self.config.d_model,
                num_heads=self.config.num_heads,
                ff_dim=self.config.ff_dim,
                dropout_rate=self.config.dropout_rate
            )(x)

        for i, units in enumerate(self.config.lstm_units):
            return_sequences = (i < len(self.config.lstm_units) - 1)

            if self.config.use_residual:
                x = ResidualLSTMBlock(
                    units=units,
                    return_sequences=return_sequences,
                    dropout_rate=self.config.dropout_rate,
                    use_batch_norm=self.config.use_batch_norm,
                    bidirectional=False
                )(x)
            else:
                x = layers.LSTM(
                    units,
                    return_sequences=return_sequences,
                    kernel_initializer=xavier_initializer,
                    recurrent_initializer=get_initializer("orthogonal", seed=42)
                )(x)

                if self.config.use_batch_norm:
                    x = layers.BatchNormalization()(x)
                x = layers.Dropout(self.config.dropout_rate)(x)

        x = layers.Dense(
            32,
            activation="relu",
            kernel_initializer=xavier_initializer,
            kernel_regularizer=tf.keras.regularizers.l2(self.config.l2_reg)
        )(x)
        x = layers.Dropout(self.config.dropout_rate)(x)

        outputs = layers.Dense(
            self.config.output_dim,
            kernel_initializer=xavier_initializer
        )(x)

        self.model = Model(inputs=inputs, outputs=outputs, name=self.name)
        return self.model
'''

# ==========================================
# 9. models/cnn_model.py
# ==========================================
FILES["models/cnn_model.py"] = '''"""
CNN模型实现
用于时序数据的1D卷积
"""
import tensorflow as tf
from tensorflow.keras import Model, layers

from models.base_model import BaseModel
from models.layers import ResidualCNNBlock
from models.initializers import get_initializer
from config.model_config import CNNConfig, DEFAULT_CNN_CONFIG


class CNNModel(BaseModel):
    """CNN模型"""

    def __init__(self, config: CNNConfig = DEFAULT_CNN_CONFIG, name: str = "cnn_model"):
        super().__init__(config, name)
        self.config = config

    def build_model(self):
        inputs = layers.Input(shape=self.config.input_shape)
        x = inputs

        xavier_initializer = get_initializer("xavier", seed=42)

        for i, (filters, kernel_size, pool_size) in enumerate(zip(
            self.config.filters_list,
            self.config.kernel_sizes,
            self.config.pool_sizes
        )):
            if self.config.use_residual:
                x = ResidualCNNBlock(
                    filters=filters,
                    kernel_size=kernel_size,
                    pool_size=pool_size,
                    dropout_rate=self.config.dropout_rate,
                    use_batch_norm=self.config.use_batch_norm
                )(x)
            else:
                x = layers.Conv1D(
                    filters,
                    kernel_size,
                    padding="same",
                    kernel_initializer=xavier_initializer,
                    kernel_regularizer=tf.keras.regularizers.l2(self.config.l2_reg)
                )(x)

                if self.config.use_batch_norm:
                    x = layers.BatchNormalization()(x)
                x = layers.Activation("relu")(x)
                x = layers.Dropout(self.config.dropout_rate)(x)

                if pool_size > 1:
                    x = layers.MaxPooling1D(pool_size)(x)

        x = layers.Flatten()(x)

        x = layers.Dense(
            32,
            activation="relu",
            kernel_initializer=xavier_initializer,
            kernel_regularizer=tf.keras.regularizers.l2(self.config.l2_reg)
        )(x)
        x = layers.Dropout(self.config.dropout_rate)(x)

        outputs = layers.Dense(
            self.config.output_dim,
            kernel_initializer=xavier_initializer
        )(x)

        self.model = Model(inputs=inputs, outputs=outputs, name=self.name)
        return self.model
'''

# ==========================================
# 10. models/mlp_model.py
# ==========================================
FILES["models/mlp_model.py"] = '''"""
MLP模型实现
简单的多层感知机，用于基线对比
"""
import tensorflow as tf
from tensorflow.keras import Model, layers

from models.base_model import BaseModel
from models.layers import ResidualBlock
from models.initializers import get_initializer
from config.model_config import MLPConfig, DEFAULT_MLP_CONFIG


class MLPModel(BaseModel):
    """MLP模型"""

    def __init__(self, config: MLPConfig = DEFAULT_MLP_CONFIG, name: str = "mlp_model"):
        super().__init__(config, name)
        self.config = config

    def build_model(self):
        inputs = layers.Input(shape=self.config.input_shape)
        x = layers.Flatten()(inputs)

        xavier_initializer = get_initializer("xavier", seed=42)

        for i, units in enumerate(self.config.hidden_units):
            if self.config.use_residual:
                x = ResidualBlock(
                    units=units,
                    activation="relu",
                    dropout_rate=self.config.dropout_rate,
                    use_batch_norm=self.config.use_batch_norm
                )(x)
            else:
                x = layers.Dense(
                    units,
                    kernel_initializer=xavier_initializer,
                    kernel_regularizer=tf.keras.regularizers.l2(self.config.l2_reg)
                )(x)

                if self.config.use_batch_norm:
                    x = layers.BatchNormalization()(x)
                x = layers.Activation("relu")(x)
                x = layers.Dropout(self.config.dropout_rate)(x)

        x = layers.Dense(
            32,
            activation="relu",
            kernel_initializer=xavier_initializer,
            kernel_regularizer=tf.keras.regularizers.l2(self.config.l2_reg)
        )(x)
        x = layers.Dropout(self.config.dropout_rate)(x)

        outputs = layers.Dense(
            self.config.output_dim,
            kernel_initializer=xavier_initializer
        )(x)

        self.model = Model(inputs=inputs, outputs=outputs, name=self.name)
        return self.model
'''

# ==========================================
# 11. models/ensemble_model.py
# ==========================================
FILES["models/ensemble_model.py"] = '''"""
集成模型
支持加权平均和堆叠集成
"""
import tensorflow as tf
from tensorflow.keras import Model, layers
import numpy as np
from typing import List

from models.base_model import BaseModel
from models.initializers import get_initializer
from config.model_config import EnsembleConfig, DEFAULT_ENSEMBLE_CONFIG


class EnsembleModel(BaseModel):
    """多模型集成"""

    def __init__(
        self,
        base_models: List[BaseModel],
        config: EnsembleConfig = DEFAULT_ENSEMBLE_CONFIG,
        name: str = "ensemble_model"
    ):
        super().__init__(config, name)
        self.base_models = base_models
        self.config = config

    def build_model(self):
        if len(self.base_models) == 0:
            raise ValueError("No base models provided")

        for model in self.base_models:
            if model.model is None:
                model.build_model()

        inputs = layers.Input(shape=self.base_models[0].config.input_shape)

        predictions = []
        for i, model in enumerate(self.base_models):
            pred = model.model(inputs)
            predictions.append(pred)

        if len(predictions) == 1:
            outputs = predictions[0]
        else:
            if self.config.use_weighted_average:
                weights = self.config.weights
                if weights is None:
                    weights = [1.0 / len(predictions)] * len(predictions)

                weighted_preds = []
                for pred, weight in zip(predictions, weights):
                    weighted_pred = layers.Lambda(lambda x: x * weight)(pred)
                    weighted_preds.append(weighted_pred)

                outputs = layers.Add()(weighted_preds)
            elif self.config.use_stacking:
                concatenated = layers.Concatenate()(predictions)
                x = layers.Dense(
                    self.config.stacking_units,
                    activation="relu",
                    kernel_initializer=get_initializer("xavier", seed=42)
                )(concatenated)
                outputs = layers.Dense(
                    self.config.output_dim,
                    kernel_initializer=get_initializer("xavier", seed=42)
                )(x)
            else:
                outputs = layers.Average()(predictions)

        self.model = Model(inputs=inputs, outputs=outputs, name=self.name)
        return self.model

    def predict_ensemble(self, X: np.ndarray) -> np.ndarray:
        predictions = []
        for model in self.base_models:
            pred = model.predict(X)
            predictions.append(pred)
        return np.array(predictions)
'''

# ==========================================
# 12. models/trainer.py
# ==========================================
FILES["models/trainer.py"] = '''"""
训练器类
用于统一训练和评估所有模型
"""
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
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

    def __init__(self, training_config: TrainingConfig = DEFAULT_TRAINING_CONFIG):
        self.training_config = training_config
        self.models: Dict[str, BaseModel] = {}
        self.histories: Dict[str, Dict] = {}
        self.results: Dict[str, Dict] = {}

    def create_all_models(self, input_shape: tuple = (20, 33)):
        lstm_config = DEFAULT_LSTM_CONFIG
        lstm_config.input_shape = input_shape
        self.models["lstm"] = LSTMModel(lstm_config)

        transformer_config = DEFAULT_TRANSFORMER_LSTM_CONFIG
        transformer_config.input_shape = input_shape
        self.models["transformer_lstm"] = TransformerLSTMModel(transformer_config)

        cnn_config = DEFAULT_CNN_CONFIG
        cnn_config.input_shape = input_shape
        self.models["cnn"] = CNNModel(cnn_config)

        mlp_config = DEFAULT_MLP_CONFIG
        mlp_config.input_shape = input_shape
        self.models["mlp"] = MLPModel(mlp_config)

    def train_all(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray
    ):
        for name, model in self.models.items():
            print(f"\\n{'='*60}")
            print(f"Training {name} model...")
            print('='*60)

            model.build_model()
            model.summary()

            history = model.train(X_train, y_train, X_val, y_val, self.training_config)
            self.histories[name] = history

        print("\\nAll models trained!")

    def evaluate_all(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        print(f"\\n{'='*60}")
        print("Evaluating all models...")
        print('='*60)

        for name, model in self.models.items():
            metrics = model.evaluate(X_test, y_test)
            self.results[name] = metrics
            print(f"\\n{name}:")
            for metric_name, value in metrics.items():
                print(f"  {metric_name}: {value:.6f}")

        return self.results

    def create_ensemble(self, model_names: Optional[List[str]] = None) -> EnsembleModel:
        if model_names is None:
            model_names = list(self.models.keys())

        base_models = [self.models[name] for name in model_names]
        ensemble = EnsembleModel(base_models)
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
            print(f"\\n{'='*60}")
            print(f"Model: {name}")
            print('='*60)
            model.summary()
'''

# ==========================================
# 13. scripts/train_models.py
# ==========================================
FILES["scripts/train_models.py"] = '''"""
训练模型脚本
运行完整的模型训练和评估流程
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import warnings
warnings.filterwarnings("ignore")

from scripts.run_data_pipeline import run_pipeline
from models.trainer import ModelTrainer
from config.model_config import DEFAULT_TRAINING_CONFIG


def prepare_data():
    print("Running data pipeline...")
    results = run_pipeline(use_sample_data=True, save_data=False)
    splits = results["splits"]

    X_train = splits["X_train"]
    y_train = splits["y_train"]
    X_val = splits["X_val"]
    y_val = splits["y_val"]
    X_test = splits["X_test"]
    y_test = splits["y_test"]

    print(f"\\nData shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  X_val:   {X_val.shape}")
    print(f"  y_val:   {y_val.shape}")
    print(f"  X_test:  {X_test.shape}")
    print(f"  y_test:  {y_test.shape}")

    return X_train, y_train, X_val, y_val, X_test, y_test


def main():
    print(f"\\n{'='*60}")
    print(f"Quant Trading Model Training")
    print('='*60)

    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data()

    trainer = ModelTrainer(DEFAULT_TRAINING_CONFIG)
    trainer.create_all_models(input_shape=X_train.shape[1:])

    trainer.summary()
    trainer.train_all(X_train, y_train, X_val, y_val)
    trainer.evaluate_all(X_test, y_test)

    print(f"\\n{'='*60}")
    print("Creating ensemble model...")
    print('='*60)
    trainer.create_ensemble()

    ensemble_metrics = trainer.models["ensemble"].evaluate(X_test, y_test)
    print("\\nEnsemble performance:")
    for metric_name, value in ensemble_metrics.items():
        print(f"  {metric_name}: {value:.6f}")

    trainer.save_results(str(PROJECT_ROOT / "results"))

    print(f"\\n{'='*60}")
    print("Training complete!")
    print('='*60)


if __name__ == "__main__":
    main()
'''


def create_files():
    """创建所有文件"""
    print("=" * 60)
    print("Creating model files...")
    print("=" * 60)

    created_count = 0

    for filepath, content in FILES.items():
        full_path = PROJECT_ROOT / filepath
        full_path.parent.mkdir(parents=True, exist_ok=True)

        with open(full_path, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"✓ Created: {filepath}")
        created_count += 1

    print(f"\n{'=' * 60}")
    print(f"Successfully created {created_count} files!")
    print('=' * 60)
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Run training: python scripts/train_models.py")
    print("=" * 60)


if __name__ == "__main__":
    create_files()