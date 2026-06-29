# CSV输出格式说明

## 输出文件结构

运行流水线后，会在 `output/` 目录生成以下CSV文件：

```
output/
├── processed_data.csv              # 完整的处理后数据
├── processed_data.parquet          # 同上，Parquet格式（更高效）
├── selected_features.csv           # 选中的特征列表
├── feature_importance.csv          # 特征重要性排序
└── sequences/
    ├── *.npy                       # numpy格式（原始二进制）
    ├── indices_*.csv               # 时间索引
    └── csv_format/                 # CSV格式序列数据
        ├── X_train.csv             # 训练集输入
        ├── y_train.csv             # 训练集目标
        ├── X_val.csv               # 验证集输入
        ├── y_val.csv               # 验证集目标
        ├── X_test.csv              # 测试集输入
        ├── y_test.csv              # 测试集目标
        ├── indices_*.csv           # 各数据集时间索引
        └── feature_names.csv       # 特征名称列表
```

## 文件详解

### 1. processed_data.csv

包含所有特征和目标变量的完整数据，每一行代表一个时间步。

**列说明：**
- `datetime`: 时间戳（索引）
- `open`, `high`, `low`, `close`, `volume`: 原始OHLCV数据
- `MA*`, `EMA*`, `MACD*`, `RSI`, `KDJ` 等: 技术指标
- `Return_*`, `LogReturn_*`: 收益率特征
- `Close_to_*`: 相对位置特征
- `Future_Return`: 未来收益率（用于验证）
- `Target_Class`: 涨跌标签（0=下跌, 1=上涨）
- `Target_Reg`: 未来收益率（回归目标）

### 2. selected_features.csv

简单的特征列表，一行一个特征名称。

### 3. feature_importance.csv

特征重要性排序，包含两列：
- `Feature`: 特征名称
- `Correlation`: 与目标变量的相关性绝对值

### 4. sequences/csv_format/ 目录下的文件

#### X_train.csv, X_val.csv, X_test.csv

**格式说明：**
- 行：每个样本
- 列：`{特征名}_t{时间步}`

**3D数组展平方式：**
```
原始形状: (样本数, 20个时间步, 33个特征)
CSV形状:  (样本数, 660列)  # 20 * 33 = 660
```

**列名示例：**
```
Close_to_Open_t0    # t=0 时刻的 Close_to_Open 特征
Close_to_Open_t1    # t=1 时刻的 Close_to_Open 特征
...
Close_to_Open_t19   # t=19 时刻的 Close_to_Open 特征
Close_to_Low_t0     # t=0 时刻的 Close_to_Low 特征
...
```

**还原3D数组（Python示例）：**
```python
import pandas as pd
import numpy as np

# 读取CSV
df_X = pd.read_csv('output/sequences/csv_format/X_train.csv')

# 获取特征名称
feature_names = pd.read_csv('output/sequences/csv_format/feature_names.csv')['feature'].tolist()
n_timesteps = 20
n_features = len(feature_names)

# 还原为3D数组: (n_samples, n_timesteps, n_features)
X = df_X.values.reshape(-1, n_timesteps, n_features)
```

#### y_train.csv, y_val.csv, y_test.csv

目标变量，一列：
- `target`: 未来收益率（回归目标）

#### indices_train.csv, indices_val.csv, indices_test.csv

每个样本对应的时间索引：
- `datetime`: 序列结束时刻的时间戳

#### feature_names.csv

特征名称列表，用于还原3D数组时的列对应关系。

## 使用示例

### 示例1：只用CSV文件进行简单机器学习

```python
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# 读取数据（使用展平的CSV）
X_train = pd.read_csv('output/sequences/csv_format/X_train.csv').values
y_train = pd.read_csv('output/sequences/csv_format/y_train.csv').values.ravel()
X_test = pd.read_csv('output/sequences/csv_format/X_test.csv').values
y_test = pd.read_csv('output/sequences/csv_format/y_test.csv').values.ravel()

# 训练模型
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)
```

### 示例2：读取完整数据进行分析

```python
import pandas as pd

# 读取完整数据
df = pd.read_csv('output/processed_data.csv', parse_dates=['datetime'], index_col='datetime')

# 查看价格走势
print(df[['close', 'MA20']].head())

# 读取特征重要性
feature_imp = pd.read_csv('output/feature_importance.csv')
print('Top 10 features:')
print(feature_imp.head(10))
```

## 格式对比

| 格式 | 优点 | 缺点 | 适用场景 |
|------|------|------|----------|
| **CSV** | 通用、可读、Excel可打开 | 文件大、加载慢 | 数据交换、查看、简单分析 |
| **Parquet** | 压缩率高、加载快、保持类型 | 需要专门库读取 | 生产环境、大数据 |
| **NPY** | Python原生、最快 | 仅Python可用 | Python模型训练 |

## 控制输出格式

可以在调用 `run_pipeline()` 时指定输出格式：

```python
from scripts.run_data_pipeline import run_pipeline

# 只保存CSV（不保存Parquet）
results = run_pipeline(
    use_sample_data=True,
    save_data=True,
    save_csv=True,       # 保存CSV
    save_parquet=False   # 不保存Parquet
)

# 只保存Parquet（更高效）
results = run_pipeline(
    use_sample_data=True,
    save_data=True,
    save_csv=False,
    save_parquet=True
)
```
