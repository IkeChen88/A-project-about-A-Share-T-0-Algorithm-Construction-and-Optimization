"""
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
        self.projection = None

    def build(self, input_shape):
        if input_shape[-1] != self.units:
            self.projection = layers.Dense(self.units, use_bias=False)
        super().build(input_shape)

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = self.dense1(inputs)
        if self.use_batch_norm:
            x = self.batchnorm(x, training=training)
        x = layers.Activation(self.activation)(x)
        x = self.dropout(x, training=training)
        x = self.dense2(x)

        residual = inputs
        if self.projection is not None:
            residual = self.projection(residual)

        return self.layernorm(residual + x)

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
        self.projection = None  # 会在build中设置

        self.output_units = output_units

    def build(self, input_shape):
        if input_shape[-1] != self.output_units:
            self.projection = layers.Dense(self.output_units, use_bias=False)
        super().build(input_shape)

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = self.lstm(inputs, training=training)
        if self.use_batch_norm:
            x = self.batchnorm(x, training=training)
        x = self.dropout(x, training=training)

        residual = inputs
        if self.projection is not None:
            residual = self.projection(residual)

        if not self.return_sequences and len(residual.shape) == 3:
            residual = residual[:, -1, :]

        return self.layernorm(residual + x)

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
        self.projection = None

    def build(self, input_shape):
        if input_shape[-1] != self.filters:
            self.projection = layers.Conv1D(self.filters, 1, padding="same", use_bias=False)
        super().build(input_shape)

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = self.conv1(inputs)
        if self.use_batch_norm:
            x = self.batchnorm(x, training=training)
        x = layers.Activation("relu")(x)
        x = self.conv2(x)

        residual = inputs
        if self.projection is not None:
            residual = self.projection(residual)

        x = self.layernorm(residual + x)

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
