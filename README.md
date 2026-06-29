# A project about A Share T-0 Algorithm Construction and Optimization
   # 中芯国际(688981)量化交易模型 - 完整流程使用指南

## 概述

本项目已支持真实数据接入，提供完整的训练与分析流程：
- 数据获取（akshare）
- 特征工程与选择
- LSTM模型训练
- SHAP可解释性分析

## 快速开始

### 方式一：使用已处理好的数据（推荐，最快）

直接运行完整流程脚本，使用现有的数据：

```bash
cd /Users/mac/PyCharmMiscProject/quant\ train\ 1
source ../.venv/bin/activate
python scripts/run_complete_pipeline.py
```

这个脚本会：
1. 加载已处理好的数据
2. 训练LSTM模型
3. 进行特征重要性分析
4. 尝试SHAP分析（如果已安装）
5. 保存所有结果

### 方式二：接入真实数据（如果需要）

如果需要下载最新的中芯国际数据：

```bash
# 首先安装数据源库
pip install akshare

# 运行完整数据获取与分析流程
python scripts/complete_pipeline.py --stock_code 688981 --data_source auto
```

## 项目结构

```
quant train 1/
├── config/
│   ├── settings.py                  # 全局配置（股票代码、时间周期等）
│   └── model_config.py              # 模型配置 dataclass（LSTM/Transformer/CNN/MLP/Ensemble）
├── data/
│   ├── data_loader.py               # 数据加载
│   ├── data_helper.py               # 数据辅助（加载/标准化）
│   ├── indicators.py                # 技术指标计算
│   ├── feature_engineering.py       # 特征工程
│   ├── feature_selection.py         # 特征选择
│   └── time_series.py               # 时间序列构建
├── models/
│   ├── base_model.py                # 基类：编译/训练/评估/早停/保存
│   ├── lstm_model.py                # LSTM 模型
│   ├── transformer_lstm.py          # Transformer-LSTM 混合模型
│   ├── cnn_model.py                 # CNN 模型
│   ├── mlp_model.py                 # MLP 基线模型
│   ├── ensemble_model.py            # 集成模型（加权/Stacking/可学习权重）
│   ├── layers.py                    # 自定义层（自注意力/位置编码/残差块/可学习集成权重）
│   ├── initializers.py              # 权重初始化器
│   ├── trainer.py                   # 模型训练器（多模型管理 + Optuna 自动选模）
│   ├── ts_cross_validation.py       # 时间序列 Walk-Forward 交叉验证
│   ├── hyperparameter_optimizer.py  # Optuna 超参数优化（支持 4 种模型类型）
│   ├── explainability.py            # 模型可解释性（SHAP）
│   └── explainability_visualization.py
├── scripts/
│   ├── train_final.py               # 完整训练脚本（支持 --model_type / --compare_models / --use_ensemble）
│   ├── complete_pipeline.py         # 端到端流水线（数据→特征→训练→SHAP）
│   ├── quick_train.py               # 快速训练
│   └── simple_train.py              # 最简训练
├── output/                          # 预处理好的数据
│   ├── processed_data.parquet
│   ├── selected_features.csv
│   └── sequences/                   # X_train/y_train/X_val/y_val/X_test/y_test.npy
└── requirements.txt
```
```

### 特征工程

项目包含完整的技术指标：

**趋势指标：**
- MA均线（5/10/20/60周期）
- EMA指数移动平均
- MACD指标
- BOLL布林带

**动量指标：**
- RSI相对强弱指标
- KDJ指标
- CCI指标
- ROC变动率

**成交量指标：**
- OBV能量潮
- VWAP成交量加权平均价

**波动率指标：**
- ATR平均真实波幅
- 历史波动率

### 模型配置

## 模型架构

| 模型 | 特点 |
|------|------|
| **LSTM** | 残差连接 + 批归一化 + Dropout，支持双向 LSTM |
| **Transformer-LSTM** | 多头自注意力 + 位置编码 + LSTM 时序建模 |
| **CNN** | 1D 卷积 + 残差 CNN 块，用于时序特征提取 |
| **MLP** | 残差全连接块，作为基线对比 |
| **Ensemble** | 加权平均 / Stacking / 可学习权重集成 |

### 共同特性

- **残差连接**（ResidualLSTMBlock / ResidualBlock / ResidualCNNBlock）
- **批量归一化**（BatchNormalization / LayerNormalization）
- **Dropout 防过拟合**
- **Xavier (GlorotUniform) + 正交 (Orthogonal) 初始化**
- **BCE 损失函数 + Adam 优化器**
- **早停机制**，监控验证集 AUC
- **学习率自适应衰减**（ReduceLROnPlateau）

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 快速训练 LSTM
python scripts/train_final.py

# 训练 Transformer-LSTM + Optuna 优化
python scripts/train_final.py --model_type transformer_lstm --optimize --n_trials 20

# 对比所有模型（自动 Optuna 优化 + 训练 + 评估）
python scripts/train_final.py --compare_models --n_trials 30

# 使用可学习权重的集成模型
python scripts/train_final.py --use_ensemble
```
## 使用指南

### 1. 数据准备

```python
from data.data_helper import load_processed_data, normalize_data

data = load_processed_data()
X_train, X_val, X_test = normalize_data(data['X_train'], data['X_val'], data['X_test'])
y_train, y_val, y_test = data['y_train'], data['y_val'], data['y_test']
```

### 2. 训练单个模型

```python
from models.lstm_model import LSTMModel
from models.transformer_lstm import TransformerLSTMModel
from config.model_config import LSTMConfig, TransformerLSTMConfig, TrainingConfig

# LSTM
config = LSTMConfig(input_shape=(20, 33), lstm_units=[64, 32], dropout_rate=0.3)
model = LSTMModel(config, task_type='classification')
model.compile()
model.train(X_train, y_train, X_val, y_val)

# Transformer-LSTM
config = TransformerLSTMConfig(input_shape=(20, 33), d_model=64, num_heads=4,
                                ff_dim=128, num_transformer_blocks=2, lstm_units=[32])
model = TransformerLSTMModel(config, task_type='classification')
model.compile()
model.train(X_train, y_train, X_val, y_val)

# 评估
metrics = model.evaluate(X_test, y_test)
print(f"AUC: {metrics['auc']:.4f}")
```

### 3. Optuna 超参数优化

```python
from models.hyperparameter_optimizer import OptunaOptimizer

# 支持 model_type: "lstm" | "transformer_lstm" | "cnn" | "mlp"
optimizer = OptunaOptimizer(
    X_train, y_train, X_val, y_val,
    input_shape=(20, 33),
    model_type='transformer_lstm',  # 指定模型类型
    direction='maximize',
)

best_params = optimizer.optimize(n_trials=50)
best_config = optimizer.get_best_model_config()
```

Optuna 搜索的超参数：

| 模型 | 搜索的超参数 |
|------|------------|
| LSTM | `lstm_units`, `n_layers`, `bidirectional`, `dropout_rate`, `l2_reg`, `learning_rate`, `batch_size` |
| Transformer-LSTM | `d_model`, `num_heads`, `ff_dim`, `num_transformer_blocks`, `n_lstm_layers`, `lstm_units`, + 共享参数 |
| CNN | `filters`, `kernel_sizes`, `pool_sizes`, `n_conv_layers`, + 共享参数 |
| MLP | `hidden_units`, `n_hidden_layers`, + 共享参数 |

### 4. 多模型自动对比

```python
from models.trainer import ModelTrainer

trainer = ModelTrainer(task_type='classification')
comparison = trainer.compare_models(
    X_train, y_train, X_val, y_val,
    X_test=X_test, y_test=y_test,
    n_trials=30,  # 每种模型的 Optuna 试验次数
)

print(f"最佳模型: {trainer.best_model_name}")
print(f"最佳 AUC: {trainer.best_model_score['auc']:.4f}")
```

输出对比表：

```
Model              AUC       Loss     Precision  Recall
lstm               0.7234    0.5432   0.6812     0.6543
transformer_lstm   0.7512    0.5211   0.7034     0.6891  ← BEST
cnn                0.6987    0.5623   0.6543     0.6321
mlp                0.6876    0.5789   0.6432     0.6211
ensemble           0.7634    0.5102   0.7211     0.7012
```

### 5. 集成模型（可学习权重）

```python
from models.ensemble_model import EnsembleModel
from config.model_config import EnsembleConfig

# 可学习权重集成
config = EnsembleConfig(use_weighted_average=True, learnable_weights=True)
ensemble = EnsembleModel([lstm_model, transformer_model], config=config)
ensemble.compile(learning_rate=1e-4)
ensemble.train(X_train, y_train, X_val, y_val)

# 查看学习到的权重
weights = ensemble.get_learned_weights()
print(f"LSTM 权重: {weights[0]:.4f}")
print(f"Transformer-LSTM 权重: {weights[1]:.4f}")
```

### 6. 时间序列交叉验证

```python
from models.ts_cross_validation import TimeSeriesCrossValidator

cv = TimeSeriesCrossValidator(n_splits=5, task_type='classification')
results = cv.cross_validate(model_builder, X_train, y_train, X_val, y_val)

print(f"AUC Mean: {results['summary']['auc_mean']:.4f}")
print(f"AUC Std:  {results['summary']['auc_std']:.4f}")
```

## 命令行参数

```bash
python scripts/train_final.py [OPTIONS]

选项:
  --model_type {lstm,transformer_lstm,cnn,mlp,all}
                        模型类型（默认: lstm）
  --optimize            启用 Optuna 超参数优化
  --n_trials N          Optuna 试验次数（默认: 20）
  --compare_models      对比所有模型类型（自动优化+训练+评估+集成）
  --use_ensemble        使用可学习权重的集成模型
  --cross_validate      启用时间序列交叉验证
  --n_splits N          交叉验证折数（默认: 5）
  --output_dir DIR      输出目录（默认: results）
```

## 配置

编辑 `config/settings.py`：

```python
STOCK_CODE = "688981"      # 股票代码
TIMEFRAME = "5m"           # K 线周期
SEQUENCE_LENGTH = 20       # 输入序列长度（20 根 K 线）
TARGET_PERIOD = 1          # 预测未来 1 个周期
TRAIN_RATIO = 0.7          # 训练集比例
```

编辑 `config/model_config.py` 修改各模型的默认超参数。

## 依赖

```
tensorflow>=2.13.0      # 深度学习框架
optuna>=3.3.0           # 超参数优化
shap>=0.40.0            # 模型可解释性
scikit-learn>=1.3.0     # 机器学习工具
pandas>=2.0.0           # 数据处理
numpy>=1.24.0           # 数值计算
matplotlib>=3.7.0       # 可视化
plotly>=5.14.0          # 交互式可视化
```## 使用示例

### 示例1：快速运行（使用现有数据）

```python
cd /Users/mac/PyCharmMiscProject/quant\ train\ 1
source ../.venv/bin/activate
python scripts/run_complete_pipeline.py
```

### 示例2：使用真实数据

```bash
# 1. 安装akshare
pip install akshare

# 2. 运行完整流程
python scripts/complete_pipeline.py --stock_code 688981 --data_source akshare --output_dir smic_results
```

### 示例3：使用模拟数据（网络问题时）

```bash
python scripts/complete_pipeline.py --data_source sample --output_dir sample_results
```

## 配置修改

### 修改全局配置

编辑 `config/settings.py` 可以调整：
- 股票代码
- 时间周期
- 序列长度
- 训练/验证/测试比例
- 特征选择方法

### 修改模型配置

编辑 `config/model_config.py` 或在代码中直接创建配置对象。

## 常见问题

### Q1: 网络问题导致无法获取真实数据？

A: 使用模拟数据模式：
```bash
python scripts/complete_pipeline.py --data_source sample
```

或者直接使用已有的数据：
```bash
python scripts/run_complete_pipeline.py
```

### Q2: SHAP运行很慢？

A: SHAP分析确实需要较多时间。`run_complete_pipeline.py` 脚本中，即使SHAP失败，其他分析仍会继续进行。

或者可以先不安装SHAP，只使用梯度方法进行特征重要性分析。

### Q3: 如何安装SHAP？

A:
```bash
pip install shap
```

### Q4: 内存不足？

A: 减少序列长度或样本量，或使用更小的模型。

## 下一步建议

### 1. 模型优化
- 调整超参数（使用Optuna）
- 尝试不同模型架构
- 增加更多特征

### 2. 策略开发
- 将模型预测转化为交易信号
- 加入止损止盈
- 测试不同持仓周期

### 3. 风险管理
- 计算仓位大小
- 设置最大回撤限制
- 分散投资

## 技术支持

如遇问题，请检查：
1. 虚拟环境已激活
2. 依赖包已安装
3. 有足够的磁盘空间和内存


祝使用愉快！
