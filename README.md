# A-project-about-A-Share-T-0-Algorithm-Construction-and-Optimization
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

## 详细说明

### 文件说明

| 文件 | 说明 |
|------|------|
| `data/data_source.py` | 真实数据源获取模块 |
| `data/data_loader.py` | 数据加载器（已更新） |
| `scripts/complete_pipeline.py` | 完整流程脚本（包含数据获取） |
| `scripts/run_complete_pipeline.py` | 简化版脚本（使用现有数据，推荐） |

### 输出结果

运行完成后，在输出目录会生成：

```
complete_results/
├── 分析报告.txt                # 完整分析报告
├── evaluation_metrics.csv     # 模型评估指标
├── test_predictions.csv       # 测试集预测结果
├── feature_importance.csv     # 特征重要性
├── time_step_importance.csv   # 时间步重要性
├── feature_names.csv          # 特征列表
├── model.h5                   # 训练好的模型
├── shap_feature_importance.csv  # SHAP特征重要性（如果运行）
└── shap_values.npy            # SHAP值（如果运行）
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

默认使用的LSTM配置：
- 双层LSTM（32 + 16单元）
- 残差连接
- 批量归一化
- Dropout 0.3
- Adam优化器（学习率 1e-3）

## 使用示例

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

## 文件速查

```
quant train 1/
├── config/
│   ├── settings.py              # 全局配置
│   └── model_config.py          # 模型配置
├── data/
│   ├── data_source.py           # 数据源（新增）
│   ├── data_loader.py           # 数据加载（已更新）
│   ├── indicators.py            # 技术指标
│   ├── feature_engineering.py   # 特征工程
│   ├── feature_selection.py     # 特征选择
│   └── time_series.py           # 时间序列构建
├── models/
│   ├── lstm_model.py            # LSTM模型
│   ├── base_model.py            # 基础模型
│   └── explainability.py        # 可解释性分析
├── scripts/
│   ├── run_complete_pipeline.py  # 简化版完整流程（推荐）
│   └── complete_pipeline.py     # 完整版（含数据获取）
├── output/                      # 已处理好的数据
└── requirements.txt             # 依赖包
```

祝使用愉快！
