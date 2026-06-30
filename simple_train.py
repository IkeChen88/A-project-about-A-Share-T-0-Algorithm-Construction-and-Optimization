"""
简单训练脚本
用于验证模型训练功能
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import tensorflow as tf

# 设置随机种子确保可复现
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

from data.data_helper import load_processed_data, normalize_data
from models.lstm_model import LSTMModel
from config.model_config import LSTMConfig, TrainingConfig


def main():
    """简单训练"""
    print(f"\n{'='*60}")
    print(f"简单模型训练 - 验证系统功能")
    print(f"{'='*60}")

    # 1. 加载和预处理数据
    print(f"\n[步骤 1] 加载数据...")
    data = load_processed_data()

    # 标准化数据
    X_train, X_val, X_test = normalize_data(
        data['X_train'], data['X_val'], data['X_test']
    )
    y_train, y_val, y_test = data['y_train'], data['y_val'], data['y_test']

    print(f"数据形状:")
    print(f"  X_train: {X_train.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  X_val:   {X_val.shape}")
    print(f"  y_val:   {y_val.shape}")
    print(f"  X_test:  {X_test.shape}")
    print(f"  y_test:  {y_test.shape}")

    # 2. 创建模型配置
    print(f"\n[步骤 2] 创建模型配置...")
    config = LSTMConfig(
        input_shape=data['input_shape'],
        lstm_units=[32, 16],  # 小模型用于快速测试
        dropout_rate=0.3,
        l2_reg=1e-4,
        learning_rate=1e-3,
        bidirectional=False,
        use_residual=True,
        use_batch_norm=True
    )

    training_config = TrainingConfig(
        batch_size=64,
        epochs=5,  # 少epoch用于快速测试
        early_stopping_patience=3,
        reduce_lr_patience=2,
        save_dir=None
    )

    # 3. 创建和编译模型
    print(f"\n[步骤 3] 创建模型...")
    model = LSTMModel(config, task_type="classification", name="lstm_simple")
    model.compile()
    model.summary()

    # 4. 训练模型
    print(f"\n[步骤 4] 训练模型...")
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        training_config=training_config,
        verbose=1
    )

    # 5. 评估模型
    print(f"\n[步骤 5] 评估模型...")
    train_metrics = model.evaluate(X_train, y_train)
    val_metrics = model.evaluate(X_val, y_val)
    test_metrics = model.evaluate(X_test, y_test)

    print(f"\n模型评估结果:")
    print(f"  训练集:")
    for k, v in train_metrics.items():
        print(f"    {k}: {v:.6f}")
    print(f"  验证集:")
    for k, v in val_metrics.items():
        print(f"    {k}: {v:.6f}")
    print(f"  测试集:")
    for k, v in test_metrics.items():
        print(f"    {k}: {v:.6f}")

    # 6. 简单预测
    print(f"\n[步骤 6] 简单预测示例...")
    sample_pred = model.predict_proba(X_test[:5])
    print(f"  预测概率: {sample_pred.flatten()}")
    print(f"  真实标签: {y_test[:5]}")

    print(f"\n{'='*60}")
    print(f"所有功能验证完成！")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
