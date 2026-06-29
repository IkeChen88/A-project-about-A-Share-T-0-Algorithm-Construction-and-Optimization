"""
项目全局配置
"""
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 数据配置
STOCK_CODE = "688981"  # 中芯国际
STOCK_NAME = "中芯国际"
TIMEFRAME = "5m"  # 5分钟K线
START_DATE = "2020-01-01"
END_DATE = "2025-12-31"

# 数据存储路径
DATA_DIR = PROJECT_ROOT / "data_storage"
DATA_DIR.mkdir(exist_ok=True)
RAW_DATA_PATH = DATA_DIR / "raw_data.parquet"
PROCESSED_DATA_PATH = DATA_DIR / "processed_data.parquet"

# 滚动时间窗口配置
SEQUENCE_LENGTH = 20  # 输入序列长度：20个5分钟K线 = 100分钟
TARGET_PERIOD = 1     # 预测未来1个周期（5分钟）
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# 技术指标配置
MA_PERIODS = [5, 10, 20, 60]
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
KDJ_N = 9
KDJ_M1 = 3
KDJ_M2 = 3
BOLL_PERIOD = 20
ATR_PERIOD = 14

# 特征工程配置
ROLLING_WINDOWS = [5, 10, 20]  # 滚动统计窗口
FEATURE_SELECTION_METHOD = "mutual_info"  # "correlation", "mutual_info", "tree"
TOP_N_FEATURES = 50

# 模型配置（预留）
RANDOM_STATE = 42
