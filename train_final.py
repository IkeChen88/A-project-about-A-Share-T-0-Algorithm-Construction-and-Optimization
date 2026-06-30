"""
完整训练脚本
包含:
1. 数据加载和预处理
2. 超参数优化（可选，支持多种模型类型）
3. 时间序列交叉验证（可选）
4. 多模型对比（可选）
5. 最终模型训练
6. 模型评估和保存
"""
import sys
from pathlib import Path
import argparse

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import tensorflow as tf

# 设置随机种子确保可复现
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# 导入模块
from data.data_helper import load_processed_data, normalize_data
from models.lstm_model import LSTMModel
from models.transformer_lstm import TransformerLSTMModel
from models.cnn_model import CNNModel
from models.mlp_model import MLPModel
from models.ensemble_model import EnsembleModel
from models.ts_cross_validation import TimeSeriesCrossValidator
from models.hyperparameter_optimizer import OptunaOptimizer
from models.trainer import ModelTrainer
from config.model_config import (
    LSTMConfig, TransformerLSTMConfig, CNNConfig, MLPConfig,
    EnsembleConfig, TrainingConfig
)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='量化交易模型训练')
    parser.add_argument(
        '--data_dir',
        type=str,
        default=None,
        help='数据目录路径'
    )
    parser.add_argument(
        '--model_type',
        type=str,
        default='lstm',
        choices=['lstm', 'transformer_lstm', 'cnn', 'mlp', 'all'],
        help='模型类型 (默认: lstm)'
    )
    parser.add_argument(
        '--optimize',
        action='store_true',
        help='是否进行超参数优化'
    )
    parser.add_argument(
        '--n_trials',
        type=int,
        default=20,
        help='超参数优化试验次数'
    )
    parser.add_argument(
        '--compare_models',
        action='store_true',
        help='对比所有模型类型（自动优化+训练+评估）'
    )
    parser.add_argument(
        '--use_ensemble',
        action='store_true',
        help='使用可学习权重的集成模型'
    )
    parser.add_argument(
        '--cross_validate',
        action='store_true',
        help='是否进行时间序列交叉验证'
    )
    parser.add_argument(
        '--n_splits',
        type=int,
        default=5,
        help='交叉验证折数'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='results',
        help='输出目录'
    )
    parser.add_argument(
        '--skip_optimization',
        action='store_true',
        help='跳过超参数优化，使用默认参数'
    )

    return parser.parse_args()


def main():
    """主训练流程"""
    args = parse_args()
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"量化交易模型训练流程")
    print(f"{'='*60}")
    print(f"输出目录: {output_dir}")

    # ==================== 1. 数据加载和预处理 ====================
    print(f"\n[步骤 1/5] 加载和预处理数据")
    print(f"{'-'*60}")

    data = load_processed_data(args.data_dir)

    # 标准化数据
    X_train, X_val, X_test = normalize_data(
        data['X_train'], data['X_val'], data['X_test'], method='standard'
    )
    y_train, y_val, y_test = data['y_train'], data['y_val'], data['y_test']

    input_shape = data['input_shape']
    print(f"数据加载完成:")
    print(f"  训练集: X={X_train.shape}, y={y_train.shape}, 正样本={y_train.sum():.0f}")
    print(f"  验证集: X={X_val.shape}, y={y_val.shape}, 正样本={y_val.sum():.0f}")
    print(f"  测试集: X={X_test.shape}, y={y_test.shape}, 正样本={y_test.sum():.0f}")

    # ==================== 模型类型映射 ====================
    MODEL_CLASS_MAP = {
        "lstm": LSTMModel,
        "transformer_lstm": TransformerLSTMModel,
        "cnn": CNNModel,
        "mlp": MLPModel,
    }
    DEFAULT_CONFIG_MAP = {
        "lstm": lambda: LSTMConfig(input_shape=input_shape, lstm_units=[64, 32], dropout_rate=0.3, l2_reg=1e-4, learning_rate=1e-3, use_residual=True, use_batch_norm=True),
        "transformer_lstm": lambda: TransformerLSTMConfig(input_shape=input_shape, d_model=64, num_heads=4, ff_dim=128, num_transformer_blocks=2, lstm_units=[32], dropout_rate=0.2, l2_reg=1e-4, learning_rate=1e-3, use_residual=True, use_batch_norm=True),
        "cnn": lambda: CNNConfig(input_shape=input_shape, filters_list=[64, 32], kernel_sizes=[3, 3], pool_sizes=[2, 1], dropout_rate=0.3, l2_reg=1e-4, learning_rate=1e-3, use_residual=True, use_batch_norm=True),
        "mlp": lambda: MLPConfig(input_shape=input_shape, hidden_units=[128, 64, 32], dropout_rate=0.3, l2_reg=1e-4, learning_rate=1e-3, use_residual=True, use_batch_norm=True),
    }

    # ==================== 多模型对比模式 ====================
    if args.compare_models:
        print(f"\n[步骤 2/5] 多模型对比（Optuna优化 + 训练 + 评估）")
        print(f"{'-'*60}")

        trainer = ModelTrainer(
            training_config=TrainingConfig(epochs=100, early_stopping_patience=15),
            task_type="classification",
        )

        comparison_df = trainer.compare_models(
            X_train, y_train, X_val, y_val,
            X_test=X_test, y_test=y_test,
            n_trials=args.n_trials,
        )

        # 保存对比结果
        comparison_df.to_csv(output_dir / "model_comparison.csv")
        print(f"\n模型对比结果已保存: {output_dir / 'model_comparison.csv'}")

        # 保存各模型
        for name, model in trainer.models.items():
            model.save(str(output_dir / f"{name}_model.h5"))
        print(f"所有模型已保存至: {output_dir}")

        print(f"\n{'='*60}")
        print(f"多模型对比完成!")
        print(f"{'='*60}")
        return

    # ==================== 单模型模式 ====================
    model_type = args.model_type
    print(f"\n模型类型: {model_type}")

    # ==================== 2. 超参数优化 ====================
    best_config = None
    best_training_config = None

    if args.optimize and not args.skip_optimization:
        print(f"\n[步骤 2/5] 超参数优化")
        print(f"{'-'*60}")

        optimizer = OptunaOptimizer(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            input_shape=input_shape,
            task_type="classification",
            direction="maximize",
            study_name=f"{model_type}_optimization",
            model_type=model_type,
        )

        best_params = optimizer.optimize(
            n_trials=args.n_trials,
            show_progress_bar=True
        )

        # 保存优化结果
        optimizer.save_study(output_dir / "optimization_results.json")
        optimizer.plot_optimization_history(str(output_dir / "optimization_history.html"))
        optimizer.plot_param_importances(str(output_dir / "param_importances.html"))

        # 获取最佳配置
        best_config = optimizer.get_best_model_config()
        best_training_config = optimizer.get_training_config()
        best_training_config.save_dir = str(output_dir / "saved_models")

        print(f"\n最佳配置:")
        if model_type == "lstm":
            print(f"  LSTM单元: {best_config.lstm_units}")
        elif model_type == "transformer_lstm":
            print(f"  d_model: {best_config.d_model}, num_heads: {best_config.num_heads}, ff_dim: {best_config.ff_dim}")
            print(f"  Transformer blocks: {best_config.num_transformer_blocks}")
            print(f"  LSTM单元: {best_config.lstm_units}")
        elif model_type == "cnn":
            print(f"  Filters: {best_config.filters_list}")
        elif model_type == "mlp":
            print(f"  Hidden units: {best_config.hidden_units}")
        print(f"  Dropout: {best_config.dropout_rate:.3f}")
        print(f"  学习率: {best_config.learning_rate:.6f}")
        print(f"  批次大小: {best_training_config.batch_size}")
    else:
        print(f"\n[步骤 2/5] 跳过超参数优化，使用默认参数")
        print(f"{'-'*60}")

        best_config = DEFAULT_CONFIG_MAP[model_type]()
        best_training_config = TrainingConfig(
            batch_size=64,
            epochs=100,
            early_stopping_patience=15,
            reduce_lr_patience=7,
            reduce_lr_factor=0.5,
            min_learning_rate=1e-7,
            save_dir=str(output_dir / "saved_models")
        )

    # ==================== 3. 时间序列交叉验证 ====================
    if args.cross_validate:
        print(f"\n[步骤 3/5] 时间序列交叉验证")
        print(f"{'-'*60}")

        # 合并训练和验证用于交叉验证
        X_cv = np.concatenate([X_train, X_val], axis=0)
        y_cv = np.concatenate([y_train, y_val], axis=0)

        # 模型构建函数
        ModelClass = MODEL_CLASS_MAP[model_type]

        def model_builder():
            model = ModelClass(best_config, task_type="classification", name=f"{model_type}_cv")
            model.compile(learning_rate=best_config.learning_rate)
            return model

        # 执行交叉验证
        cv = TimeSeriesCrossValidator(
            n_splits=args.n_splits,
            val_window_size=min(200, len(X_val)),
            fixed_size=False,
            task_type="classification"
        )

        cv_results = cv.cross_validate(
            model_builder,
            X_cv,
            y_cv,
            X_val_fixed=X_test,
            y_val_fixed=y_test,
            training_config=best_training_config,
            verbose=1
        )

        # 保存结果
        cv.save_results(output_dir / "cv_results.json")

        print(f"\n交叉验证汇总:")
        summary = cv_results['summary']
        for key, value in summary.items():
            if not key.endswith('_values'):
                print(f"  {key}: {value}")

    # ==================== 4. 最终模型训练 ====================
    print(f"\n[步骤 4/5] 最终模型训练")
    print(f"{'-'*60}")

    # 创建和编译模型
    ModelClass = MODEL_CLASS_MAP[model_type]
    final_model = ModelClass(best_config, task_type="classification", name=f"{model_type}_final")
    final_model.compile(learning_rate=best_config.learning_rate)
    final_model.summary()

    # 训练
    print(f"\n开始训练...")
    history = final_model.train(
        X_train, y_train,
        X_val, y_val,
        training_config=best_training_config,
        verbose=1
    )

    # 保存训练历史
    history_df = pd.DataFrame(history)
    history_df.to_csv(output_dir / "training_history.csv", index=False)
    print(f"训练历史已保存: {output_dir / 'training_history.csv'}")

    # ==================== 5. 模型评估和保存 ====================
    print(f"\n[步骤 5/5] 模型评估")
    print(f"{'-'*60}")

    # 在各数据集上评估
    train_metrics = final_model.evaluate(X_train, y_train)
    val_metrics = final_model.evaluate(X_val, y_val)
    test_metrics = final_model.evaluate(X_test, y_test)

    print(f"\n训练集结果:")
    for k, v in train_metrics.items():
        print(f"  {k}: {v:.6f}")

    print(f"\n验证集结果:")
    for k, v in val_metrics.items():
        print(f"  {k}: {v:.6f}")

    print(f"\n测试集结果:")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.6f}")

    # 保存评估结果
    all_metrics = {
        'train': train_metrics,
        'val': val_metrics,
        'test': test_metrics
    }

    metrics_df = pd.DataFrame(all_metrics).T
    metrics_df.to_csv(output_dir / "evaluation_results.csv")
    print(f"\n评估结果已保存: {output_dir / 'evaluation_results.csv'}")

    # 保存模型
    final_model.save(str(output_dir / f"final_{model_type}_model.h5"))
    print(f"模型已保存: {output_dir / f'final_{model_type}_model.h5'}")

    # 保存预测结果
    test_pred = final_model.predict_proba(X_test)
    pred_df = pd.DataFrame({
        'y_true': y_test.flatten(),
        'y_pred': test_pred.flatten()
    })
    pred_df.to_csv(output_dir / "test_predictions.csv", index=False)
    print(f"测试集预测已保存: {output_dir / 'test_predictions.csv'}")

    # ==================== 集成模型（可选） ====================
    if args.use_ensemble:
        print(f"\n[额外] 创建可学习权重的集成模型")
        print(f"{'-'*60}")

        # 创建基础模型（使用相同的配置）
        base_models = [
            LSTMModel(LSTMConfig(input_shape=input_shape, lstm_units=[64, 32], dropout_rate=0.3, use_residual=True, use_batch_norm=True), task_type="classification"),
            TransformerLSTMModel(TransformerLSTMConfig(input_shape=input_shape, d_model=64, num_heads=4, ff_dim=128, num_transformer_blocks=2, lstm_units=[32], dropout_rate=0.2, use_residual=True, use_batch_norm=True), task_type="classification"),
            CNNModel(CNNConfig(input_shape=input_shape, filters_list=[64, 32], kernel_sizes=[3, 3], pool_sizes=[2, 1], dropout_rate=0.3, use_residual=True, use_batch_norm=True), task_type="classification"),
            MLPModel(MLPConfig(input_shape=input_shape, hidden_units=[128, 64, 32], dropout_rate=0.3, use_residual=True, use_batch_norm=True), task_type="classification"),
        ]

        # 先训练各基础模型
        for model in base_models:
            model.compile(learning_rate=1e-3)
            model.train(X_train, y_train, X_val, y_val, training_config=best_training_config, verbose=1)

        # 创建可学习权重的集成模型
        ensemble_config = EnsembleConfig(
            use_weighted_average=True,
            learnable_weights=True,
        )
        ensemble = EnsembleModel(base_models, config=ensemble_config, task_type="classification")
        ensemble.compile(learning_rate=1e-4)  # 较低学习率用于微调权重
        ensemble.summary()

        # 训练集成模型（仅训练权重层）
        ensemble.train(X_train, y_train, X_val, y_val, training_config=TrainingConfig(epochs=30, early_stopping_patience=10), verbose=1)

        # 评估
        ensemble_metrics = ensemble.evaluate(X_test, y_test)
        print(f"\n集成模型测试集结果:")
        for k, v in ensemble_metrics.items():
            print(f"  {k}: {v:.6f}")

        # 输出学习到的权重
        learned_weights = ensemble.get_learned_weights()
        if learned_weights is not None:
            print(f"\n学习到的模型权重:")
            for i, w in enumerate(learned_weights):
                print(f"  模型{i+1}: {w:.4f}")

        ensemble.save(str(output_dir / "ensemble_model.h5"))
        print(f"集成模型已保存: {output_dir / 'ensemble_model.h5'}")

    print(f"\n{'='*60}")
    print(f"训练流程完成!")
    print(f"{'='*60}")
    print(f"所有结果保存在: {output_dir}")


if __name__ == "__main__":
    main()
