"""
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
