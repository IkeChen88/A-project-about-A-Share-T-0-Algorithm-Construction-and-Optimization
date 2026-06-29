"""
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

    print(f"\nData shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  X_val:   {X_val.shape}")
    print(f"  y_val:   {y_val.shape}")
    print(f"  X_test:  {X_test.shape}")
    print(f"  y_test:  {y_test.shape}")

    return X_train, y_train, X_val, y_val, X_test, y_test


def main():
    print(f"\n{'='*60}")
    print(f"Quant Trading Model Training")
    print('='*60)

    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data()

    trainer = ModelTrainer(DEFAULT_TRAINING_CONFIG)
    trainer.create_all_models(input_shape=X_train.shape[1:])

    trainer.summary()
    trainer.train_all(X_train, y_train, X_val, y_val)
    trainer.evaluate_all(X_test, y_test)

    print(f"\n{'='*60}")
    print("Creating ensemble model...")
    print('='*60)
    trainer.create_ensemble()

    ensemble_metrics = trainer.models["ensemble"].evaluate(X_test, y_test)
    print("\nEnsemble performance:")
    for metric_name, value in ensemble_metrics.items():
        print(f"  {metric_name}: {value:.6f}")

    trainer.save_results(str(PROJECT_ROOT / "results"))

    print(f"\n{'='*60}")
    print("Training complete!")
    print('='*60)


if __name__ == "__main__":
    main()
