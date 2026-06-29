"""
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

    def __init__(
        self,
        config: CNNConfig = DEFAULT_CNN_CONFIG,
        name: str = "cnn_model",
        task_type: str = "classification"
    ):
        super().__init__(config, name, task_type)
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

        if self.task_type == "classification":
            outputs = layers.Dense(1, activation="sigmoid", kernel_initializer=xavier_initializer)(x)
        else:
            outputs = layers.Dense(1, kernel_initializer=xavier_initializer)(x)

        self.model = Model(inputs=inputs, outputs=outputs, name=self.name)
        return self.model
