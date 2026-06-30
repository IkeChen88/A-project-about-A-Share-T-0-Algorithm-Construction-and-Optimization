"""
快速训练脚本
使用默认参数快速训练一个简单的模型用于测试
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import tensorflow as tf

# 设置随机种子
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

from data.data_helper import load_processed_data, normalize_data
from models.lstm_model import LSTMModel
from config.model_config import LSTMConfig, TrainingConfig


def main():
    """快速训练"""
    print(f"\n{'='*60}")
    print(f"快速模型训练")
    print(f"{'='*60}")

    # 输出目录
    output_dir = PROJECT_ROOT / "quick_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ==================== 1. 加载数据 ====================
    print(f"\n[步骤 1/4] 加载数据")
    data = load_processed_data()

    # 标准化数据
    X_train, X_val, X_test = normalize_data(
        data['X_train'], data['X_val'], data['X_test']
    )
    y_train, y_val, y_test = data['y_train'], data['y_val'], data['y_test']

    print(f"数据加载完成:")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_val:   {X_val.shape}, y_val:   {y_val.shape}")
    print(f"  X_test:  {X_test.shape}, y_test:  {y_test.shape}")

    # ==================== 2. 创建模型 ====================
    print(f"\n[步骤 2/4] 创建模型")

    config = LSTMConfig(
        input_shape=data['input_shape'],
        lstm_units=[64, 32],
        dropout_rate=0.3,
        l2_reg=1e-4,
        learning_rate=1e-3,
        bidirectional=False,
        use_residual=True,
        use_batch_norm=True
    )

    training_config = TrainingConfig(
        batch_size=64,
        epochs=30,  # 较少的epochs用于快速测试
        early_stopping_patience=10,
        reduce_lr_patience=5,
        reduce_lr_factor=0.5,
        min_learning_rate=1e-7,
        save_dir=str(output_dir / "saved_models")
    )

    model = LSTMModel(config, task_type="classification", name="lstm_quick")
    model.compile()
    model.summary()

    # ==================== 3. 训练模型 ====================
    print(f"\n[步骤 3/4] 训练模型")
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        training_config=training_config,
        verbose=1
    )

    # 保存训练历史
    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)

    # ==================== 4. 评估模型 ====================
    print(f"\n[步骤 4/4] 评估模型")

    train_metrics = model.evaluate(X_train, y_train)
    val_metrics = model.evaluate(X_val, y_val)
    test_metrics = model.evaluate(X_test, y_test)

    print(f"\n训练集:")
    for k, v in train_metrics.items():
        print(f"  {k}: {v:.6f}")

    print(f"\n验证集:")
    for k, v in val_metrics.items():
        print(f"  {k}: {v:.6f}")

    print(f"\n测试集:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.6f}")

    # 保存结果
    results = {
        'train': train_metrics,
        'val': val_metrics,
        'test': test_metrics
    }
    pd.DataFrame(results).T.to_csv(output_dir / "results.csv")

    # 保存模型
    model.save(str(output_dir / "model.h5"))

    # 保存预测
    test_pred = model.predict_proba(X_test)
    pred_df = pd.DataFrame({
        'y_true': y_test.flatten(),
        'y_pred': test_pred.flatten()
    })
    pred_df.to_csv(output_dir / "predictions.csv", index=False)

    print(f"\n{'='*60}")
    print(f"完成! 结果保存在: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
