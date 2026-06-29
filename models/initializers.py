"""
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
