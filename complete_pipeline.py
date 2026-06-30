"""
完整的量化交易模型训练与分析一条龙脚本
包含：数据获取 -> 特征工程 -> 模型训练 -> 预测分析 -> SHAP解释
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))

import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子确保可复现
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# 导入项目模块
from config import settings
from data.data_loader import DataLoader
from data.indicators import TechnicalIndicators
from data.feature_engineering import FeatureEngineer
from data.feature_selection import FeatureSelector
from data.time_series import TimeSeriesBuilder
from models.lstm_model import LSTMModel
from models.transformer_lstm import TransformerLSTMModel
from models.cnn_model import CNNModel
from models.mlp_model import MLPModel
from models.base_model import BaseModel
from config.model_config import (
    LSTMConfig, TransformerLSTMConfig, CNNConfig, MLPConfig,
    TrainingConfig, DEFAULT_LSTM_CONFIG, DEFAULT_TRAINING_CONFIG
)

# SHAP相关
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class CompletePipeline:
    """完整的模型训练与分析流程"""

    def __init__(
        self,
        stock_code: str = None,
        data_source: str = "auto",
        output_dir: str = "pipeline_results",
        model_type: str = "lstm",
    ):
        """
        初始化流程

        参数:
            stock_code: 股票代码
            data_source: 数据源
            output_dir: 输出目录
            model_type: 模型类型 ("lstm", "transformer_lstm", "cnn", "mlp")
        """
        self.stock_code = stock_code if stock_code else settings.STOCK_CODE
        self.data_source = data_source
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.model_type = model_type

        # 数据变量
        self.raw_data = None
        self.processed_data = None
        self.selected_features = None
        self.splits = None

        # 模型变量
        self.model = None
        self.history = None

        # 分析结果
        self.shap_values = None
        self.feature_importance = None

    def step1_load_data(self):
        """步骤1: 加载数据"""
        print("\n" + "=" * 80)
        print("步骤 1/6: 加载数据")
        print("=" * 80)

        loader = DataLoader()
        self.raw_data = loader.load_stock_data(
            source=self.data_source,
            use_cache=True,
            symbol=self.stock_code
        )

        print(f"\n数据加载完成:")
        print(f"  数据范围: {self.raw_data.index[0]} 至 {self.raw_data.index[-1]}")
        print(f"  数据条数: {len(self.raw_data)}")
        print(f"  数据列: {list(self.raw_data.columns)}")
        print(f"\n数据预览:")
        print(self.raw_data.head())

        # 保存原始数据
        self.raw_data.to_csv(self.output_dir / "01_raw_data.csv")
        self.raw_data.to_parquet(self.output_dir / "01_raw_data.parquet")

        return self.raw_data

    def step2_feature_engineering(self):
        """步骤2: 特征工程"""
        print("\n" + "=" * 80)
        print("步骤 2/6: 特征工程")
        print("=" * 80)

        # 添加技术指标
        print("\n添加技术指标...")
        indicators = TechnicalIndicators(self.raw_data)
        df_with_indicators = indicators.add_all_indicators()

        # 特征工程
        print("\n构建特征...")
        engineer = FeatureEngineer(df_with_indicators)
        df_with_features = engineer.build_all_features()

        # 创建目标变量
        print("\n创建目标变量...")
        df_with_features = engineer.create_target(df_with_features)

        self.processed_data = df_with_features

        print(f"\n特征工程完成:")
        print(f"  总列数: {len(self.processed_data.columns)}")
        print(f"  特征列数: {len([col for col in self.processed_data.columns if col not in ['Future_Return', 'Target_Class', 'Target_Reg']])}")

        # 保存数据
        self.processed_data.to_csv(self.output_dir / "02_features.csv")
        self.processed_data.to_parquet(self.output_dir / "02_features.parquet")

        return self.processed_data

    def step3_feature_selection(self):
        """步骤3: 特征选择"""
        print("\n" + "=" * 80)
        print("步骤 3/6: 特征选择")
        print("=" * 80)

        selector = FeatureSelector(self.processed_data, target_col='Target_Reg')

        # 选择特征
        print("\n选择特征...")
        self.selected_features = selector.select_features(
            method=settings.FEATURE_SELECTION_METHOD,
            top_n=settings.TOP_N_FEATURES
        )

        # 移除高度相关特征
        print("\n移除高度相关特征...")
        self.selected_features = selector.remove_highly_correlated(
            self.selected_features,
            threshold=0.95
        )

        # 获取特征重要性
        importance_df = selector.get_feature_importance(self.selected_features)

        # 过滤数据
        self.processed_data = selector.filter_features(self.processed_data, self.selected_features)

        print(f"\n特征选择完成:")
        print(f"  最终特征数: {len(self.selected_features)}")
        print(f"\nTop 10 重要特征:")
        print(importance_df.head(10))

        # 保存
        importance_df.to_csv(self.output_dir / "03_feature_importance.csv", index=False)
        pd.DataFrame({'feature': self.selected_features}).to_csv(
            self.output_dir / "03_selected_features.csv", index=False
        )
        self.processed_data.to_csv(self.output_dir / "03_selected_data.csv")
        self.processed_data.to_parquet(self.output_dir / "03_selected_data.parquet")

        self.feature_importance = importance_df

        return self.selected_features, self.processed_data

    def step4_time_series_build(self):
        """步骤4: 构建时间序列"""
        print("\n" + "=" * 80)
        print("步骤 4/6: 构建时间序列")
        print("=" * 80)

        builder = TimeSeriesBuilder(self.processed_data, self.selected_features)

        # 创建序列
        print(f"\n创建序列（窗口大小: {settings.SEQUENCE_LENGTH}）...")
        X, y, indices = builder.create_sequences(
            sequence_length=settings.SEQUENCE_LENGTH,
            target_period=settings.TARGET_PERIOD
        )

        # 划分数据集
        print("\n划分训练/验证/测试集...")
        self.splits = builder.split_train_val_test(
            X, y, indices,
            train_ratio=settings.TRAIN_RATIO,
            val_ratio=settings.VAL_RATIO
        )

        print(f"\n时间序列构建完成:")
        print(f"  X_train shape: {self.splits['X_train'].shape}")
        print(f"  y_train shape: {self.splits['y_train'].shape}")
        print(f"  X_val shape:   {self.splits['X_val'].shape}")
        print(f"  y_val shape:   {self.splits['y_val'].shape}")
        print(f"  X_test shape:  {self.splits['X_test'].shape}")
        print(f"  y_test shape:  {self.splits['y_test'].shape}")

        # 保存序列数据
        np.save(self.output_dir / "04_X_train.npy", self.splits['X_train'])
        np.save(self.output_dir / "04_y_train.npy", self.splits['y_train'])
        np.save(self.output_dir / "04_X_val.npy", self.splits['X_val'])
        np.save(self.output_dir / "04_y_val.npy", self.splits['y_val'])
        np.save(self.output_dir / "04_X_test.npy", self.splits['X_test'])
        np.save(self.output_dir / "04_y_test.npy", self.splits['y_test'])

        return self.splits

    def step5_train_model(self):
        """步骤5: 训练模型"""
        print("\n" + "=" * 80)
        print(f"步骤 5/6: 训练模型 ({self.model_type})")
        print("=" * 80)

        input_shape = self.splits['X_train'].shape[1:]

        # 根据model_type创建配置和模型
        if self.model_type == "lstm":
            model_config = LSTMConfig(
                input_shape=input_shape,
                lstm_units=[64, 32],
                dropout_rate=0.3,
                l2_reg=1e-4,
                learning_rate=1e-3,
                use_residual=True,
                use_batch_norm=True,
            )
            ModelClass = LSTMModel
        elif self.model_type == "transformer_lstm":
            model_config = TransformerLSTMConfig(
                input_shape=input_shape,
                d_model=64,
                num_heads=4,
                ff_dim=128,
                num_transformer_blocks=2,
                lstm_units=[32],
                dropout_rate=0.2,
                l2_reg=1e-4,
                learning_rate=1e-3,
                use_residual=True,
                use_batch_norm=True,
            )
            ModelClass = TransformerLSTMModel
        elif self.model_type == "cnn":
            model_config = CNNConfig(
                input_shape=input_shape,
                filters_list=[64, 32],
                kernel_sizes=[3, 3],
                pool_sizes=[2, 1],
                dropout_rate=0.3,
                l2_reg=1e-4,
                learning_rate=1e-3,
                use_residual=True,
                use_batch_norm=True,
            )
            ModelClass = CNNModel
        elif self.model_type == "mlp":
            model_config = MLPConfig(
                input_shape=input_shape,
                hidden_units=[128, 64, 32],
                dropout_rate=0.3,
                l2_reg=1e-4,
                learning_rate=1e-3,
                use_residual=True,
                use_batch_norm=True,
            )
            ModelClass = MLPModel
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

        training_config = TrainingConfig(
            batch_size=64,
            epochs=50,
            early_stopping_patience=10,
            reduce_lr_patience=5,
            save_dir=str(self.output_dir / "model_checkpoint")
        )

        # 创建并编译模型
        self.model = ModelClass(model_config, task_type="classification", name=f"smic_{self.model_type}")
        self.model.compile()

        print("\n模型结构:")
        self.model.summary()

        # 获取训练数据
        X_train = self.splits['X_train']
        y_train = (self.splits['y_train'] > 0).astype(int)  # 转换为分类目标
        X_val = self.splits['X_val']
        y_val = (self.splits['y_val'] > 0).astype(int)

        # 标准化数据
        X_train, X_val, X_test = self._normalize_data(X_train, X_val, self.splits['X_test'])
        self.splits['X_train_norm'] = X_train
        self.splits['X_val_norm'] = X_val
        self.splits['X_test_norm'] = X_test

        # 训练模型
        print("\n开始训练...")
        self.history = self.model.train(
            X_train, y_train,
            X_val, y_val,
            training_config=training_config,
            verbose=1
        )

        # 保存训练历史
        history_df = pd.DataFrame(self.history)
        history_df.to_csv(self.output_dir / "05_training_history.csv")

        # 评估模型
        print("\n评估模型...")
        train_metrics = self.model.evaluate(X_train, y_train)
        val_metrics = self.model.evaluate(X_val, y_val)
        test_metrics = self.model.evaluate(X_test, (self.splits['y_test'] > 0).astype(int))

        print("\n评估结果:")
        print(f"  训练集: {train_metrics}")
        print(f"  验证集: {val_metrics}")
        print(f"  测试集: {test_metrics}")

        # 保存评估结果
        eval_results = {
            'train': train_metrics,
            'val': val_metrics,
            'test': test_metrics
        }
        pd.DataFrame(eval_results).T.to_csv(self.output_dir / "05_evaluation_results.csv")

        # 保存模型
        self.model.save(str(self.output_dir / "05_model.h5"))

        return self.model, self.history

    def step6_shap_analysis(self):
        """步骤6: SHAP分析"""
        print("\n" + "=" * 80)
        print("步骤 6/6: SHAP分析")
        print("=" * 80)

        if not SHAP_AVAILABLE:
            print("\nSHAP未安装，跳过SHAP分析")
            print("可运行: pip install shap")
            return None

        print("\n准备SHAP分析数据...")
        # 使用测试集的子集进行SHAP分析以提高速度
        X_explain = self.splits['X_test_norm'][:100]

        # 创建特征名（包含时间步）
        feature_names_flat = []
        for t in range(X_explain.shape[1]):
            for feat in self.selected_features:
                feature_names_flat.append(f"{feat}_t{X_explain.shape[1] - t - 1}")

        # 展平数据
        X_explain_flat = X_explain.reshape(X_explain.shape[0], -1)
        X_background_flat = self.splits['X_train_norm'][:50].reshape(50, -1)

        print("\n初始化SHAP解释器...")
        # 创建模型预测函数
        def model_predict(X_flat):
            X_reshaped = X_flat.reshape(-1, X_explain.shape[1], X_explain.shape[2])
            return self.model.predict(X_reshaped, verbose=0).flatten()

        # 使用KernelSHAP
        explainer = shap.KernelExplainer(
            model_predict,
            X_background_flat
        )

        print("\n计算SHAP值...")
        self.shap_values = explainer.shap_values(X_explain_flat, nsamples=100)

        # 保存SHAP值
        np.save(self.output_dir / "06_shap_values.npy", self.shap_values)

        # 聚合特征重要性（按特征名，不区分时间步）
        print("\n计算聚合特征重要性...")
        feature_importance_shap = {}
        for feat in self.selected_features:
            feat_indices = [i for i, name in enumerate(feature_names_flat) if feat in name]
            if feat_indices:
                feature_importance_shap[feat] = np.abs(self.shap_values[:, feat_indices]).mean()

        shap_importance_df = pd.DataFrame({
            'feature': list(feature_importance_shap.keys()),
            'shap_importance': list(feature_importance_shap.values())
        }).sort_values('shap_importance', ascending=False).reset_index(drop=True)

        shap_importance_df.to_csv(self.output_dir / "06_shap_feature_importance.csv", index=False)

        print("\nSHAP特征重要性Top 10:")
        print(shap_importance_df.head(10))

        # 生成简单的可视化数据（保存到CSV，可在外部绘图）
        self._generate_shap_visualization_data(X_explain_flat, feature_names_flat)

        # 尝试生成SHAP图
        try:
            self._plot_shap_summary(explainer, self.shap_values, X_explain_flat, feature_names_flat)
        except Exception as e:
            print(f"\nSHAP可视化生成失败: {e}")
            print("SHAP数据已保存，可在后续分析中使用")

        return shap_importance_df

    def _normalize_data(self, X_train, X_val, X_test):
        """标准化数据"""
        n_train, seq_len, n_feat = X_train.shape
        X_train_flat = X_train.reshape(-1, n_feat)

        mean = X_train_flat.mean(axis=0)
        std = X_train_flat.std(axis=0)
        std = np.where(std < 1e-8, 1, std)

        X_train_flat = (X_train_flat - mean) / std
        X_val_flat = (X_val.reshape(-1, n_feat) - mean) / std
        X_test_flat = (X_test.reshape(-1, n_feat) - mean) / std

        return (
            X_train_flat.reshape(n_train, seq_len, n_feat),
            X_val_flat.reshape(X_val.shape[0], seq_len, n_feat),
            X_test_flat.reshape(X_test.shape[0], seq_len, n_feat)
        )

    def _generate_shap_visualization_data(self, X_explain, feature_names):
        """生成SHAP可视化数据"""
        # 保存Top特征的SHAP值
        shap_abs = np.abs(self.shap_values).mean(axis=0)
        top_indices = shap_abs.argsort()[-15:][::-1]

        top_features = [feature_names[i] for i in top_indices]
        top_shap_values = self.shap_values[:, top_indices]
        top_feature_values = X_explain[:, top_indices]

        # 保存
        pd.DataFrame({
            'feature': top_features,
            'mean_shap_abs': shap_abs[top_indices]
        }).to_csv(self.output_dir / "06_top_shap_features.csv", index=False)

        np.save(self.output_dir / "06_top_shap_values.npy", top_shap_values)
        np.save(self.output_dir / "06_top_feature_values.npy", top_feature_values)

    def _plot_shap_summary(self, explainer, shap_values, X_explain, feature_names):
        """尝试生成SHAP摘要图"""
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(12, 8))
            shap_values_2d = shap_values.reshape(-1, shap_values.shape[-1])
            X_explain_2d = X_explain.reshape(-1, X_explain.shape[-1])

            # 按特征聚合
            feature_shap = {}
            feature_values = {}
            for i, name in enumerate(feature_names):
                # 提取基础特征名（去掉时间步）
                base_name = name.split('_t')[0] if '_t' in name else name
                if base_name not in feature_shap:
                    feature_shap[base_name] = []
                    feature_values[base_name] = []
                feature_shap[base_name].extend(shap_values_2d[:, i].tolist())
                feature_values[base_name].extend(X_explain_2d[:, i].tolist())

            # 计算聚合重要性
            agg_importance = {k: np.abs(v).mean() for k, v in feature_shap.items()}
            top_features = sorted(agg_importance.keys(), key=agg_importance.get, reverse=True)[:15]

            # 保存聚合结果
            agg_df = pd.DataFrame({
                'feature': list(agg_importance.keys()),
                'importance': list(agg_importance.values())
            }).sort_values('importance', ascending=False)
            agg_df.to_csv(self.output_dir / "06_shap_aggregated_importance.csv", index=False)

            print(f"\nTop特征已保存，可用于后续可视化")
            return agg_df

        except Exception as e:
            print(f"SHAP图生成错误: {e}")
            return None

    def run_all(self):
        """运行完整流程"""
        print("\n" + "=" * 80)
        print("中芯国际(688981)量化交易模型训练与分析")
        print("=" * 80)

        try:
            self.step1_load_data()
            self.step2_feature_engineering()
            self.step3_feature_selection()
            self.step4_time_series_build()
            self.step5_train_model()
            self.step6_shap_analysis()

            print("\n" + "=" * 80)
            print("完整流程执行完成!")
            print("=" * 80)
            print(f"\n所有结果已保存至: {self.output_dir.absolute()}")

            self._generate_final_report()

        except Exception as e:
            print(f"\n错误发生: {e}")
            import traceback
            traceback.print_exc()

    def _generate_final_report(self):
        """生成最终报告"""
        report_file = self.output_dir / "完整分析报告.txt"

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("中芯国际(688981)量化交易模型分析报告\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"分析时间: {pd.Timestamp.now()}\n\n")

            if self.raw_data is not None:
                f.write("1. 数据情况\n")
                f.write("-" * 40 + "\n")
                f.write(f"数据范围: {self.raw_data.index[0]} 至 {self.raw_data.index[-1]}\n")
                f.write(f"数据条数: {len(self.raw_data)}\n\n")

            if self.selected_features is not None:
                f.write("2. 特征情况\n")
                f.write("-" * 40 + "\n")
                f.write(f"特征数量: {len(self.selected_features)}\n")
                f.write(f"特征列表: {self.selected_features}\n\n")

            if self.feature_importance is not None:
                f.write("3. Top 10 重要特征\n")
                f.write("-" * 40 + "\n")
                f.write(self.feature_importance.head(10).to_string())
                f.write("\n\n")

            f.write("4. 结果文件\n")
            f.write("-" * 40 + "\n")
            for f_path in sorted(self.output_dir.glob("*")):
                if f_path.is_file():
                    f.write(f"  {f_path.name}\n")

        print(f"\n最终报告已生成: {report_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='完整的量化交易模型训练与分析流程')
    parser.add_argument(
        '--stock_code',
        type=str,
        default='688981',
        help='股票代码（默认：688981）'
    )
    parser.add_argument(
        '--data_source',
        type=str,
        default='auto',
        help='数据源（auto/akshare/tushare/sample）'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='pipeline_results',
        help='输出目录'
    )

    args = parser.parse_args()

    # 创建并运行流程
    pipeline = CompletePipeline(
        stock_code=args.stock_code,
        data_source=args.data_source,
        output_dir=args.output_dir
    )

    pipeline.run_all()


if __name__ == "__main__":
    main()
