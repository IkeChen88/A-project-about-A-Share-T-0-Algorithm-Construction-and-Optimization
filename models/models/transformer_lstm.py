"""
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
