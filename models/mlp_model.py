"""
MLP模型实现
简单的多层感知机，用于基线对比
"""
import tensorflow as tf
from tensorflow.keras import Model, layers

from models.base_model import BaseModel
from models.layers import ResidualBlock
from config.model_config import MLPConfig, DEFAULT_MLP_CONFIG


class MLPModel(BaseModel):
    """MLP模型"""

    def __init__(
        self,
        config: MLPConfig = DEFAULT_MLP_CONFIG,
        name: str = "mlp_model",
        task_type: str = "classification"
    ):
        super().__init__(config, name, task_type)
        self.config = config

    def build_model(self):
        inputs = layers.Input(shape=self.config.input_shape)
        x = layers.Flatten()(inputs)

        # 使用TF内置初始化器（兼容Keras序列化）
        kernel_init = tf.keras.initializers.GlorotUniform(seed=42)

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
                    kernel_initializer=kernel_init,
                    kernel_regularizer=tf.keras.regularizers.l2(self.config.l2_reg)
                )(x)

                if self.config.use_batch_norm:
                    x = layers.BatchNormalization()(x)
                x = layers.Activation("relu")(x)
                x = layers.Dropout(self.config.dropout_rate)(x)

        x = layers.Dense(
            32,
            activation="relu",
            kernel_initializer=kernel_init,
            kernel_regularizer=tf.keras.regularizers.l2(self.config.l2_reg)
        )(x)
        x = layers.Dropout(self.config.dropout_rate)(x)

        if self.task_type == "classification":
            outputs = layers.Dense(1, activation="sigmoid", kernel_initializer=kernel_init)(x)
        else:
            outputs = layers.Dense(1, kernel_initializer=kernel_init)(x)

        self.model = Model(inputs=inputs, outputs=outputs, name=self.name)
        return self.model
