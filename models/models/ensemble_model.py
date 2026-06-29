"""
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
