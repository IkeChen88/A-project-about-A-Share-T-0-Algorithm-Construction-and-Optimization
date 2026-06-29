"""
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
