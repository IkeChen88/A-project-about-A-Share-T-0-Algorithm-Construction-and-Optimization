"""
数据加载器模块
支持从多种数据源加载股票数据
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Union
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import settings

# 导入新创建的数据源模块
try:
    from data.data_source import StockDataSource, check_data_source_available
    DATA_SOURCE_AVAILABLE = True
except ImportError:
    DATA_SOURCE_AVAILABLE = False


class DataLoader:
    """数据加载器"""

    def __init__(self):
        self.stock_code = settings.STOCK_CODE
        self.timeframe = settings.TIMEFRAME

    def load_stock_data(
        self,
        source: str = "auto",
        use_cache: bool = True,
        **kwargs
    ) -> pd.DataFrame:
        """
        加载股票数据的统一接口

        参数:
            source: 数据源选择（auto, akshare, tushare, sample, csv, parquet）
            use_cache: 是否使用缓存数据
            **kwargs: 其他参数

        返回:
            包含OHLCV数据的DataFrame
        """
        # 检查是否使用缓存
        if use_cache:
            cached_data = self._load_cached_data()
            if cached_data is not None:
                print("使用缓存数据")
                return cached_data

        # 根据source选择加载方式
        if source in ["auto", "akshare", "tushare", "sample"]:
            if DATA_SOURCE_AVAILABLE:
                df = self._load_from_data_source(source, **kwargs)
            else:
                print("数据源模块不可用，使用模拟数据")
                df = self.load_sample_data()
        elif source == "csv":
            df = self.load_from_csv(**kwargs)
        elif source == "parquet":
            df = self.load_from_parquet(**kwargs)
        else:
            raise ValueError(f"未知数据源: {source}")

        # 预处理数据
        df = self.preprocess_data(df)

        # 缓存数据
        if use_cache:
            self._save_cache(df)

        return df

    def _load_from_data_source(self, source: str, **kwargs) -> pd.DataFrame:
        """从数据源加载数据"""
        data_source = StockDataSource()
        df = data_source.get_stock_data(source=source, **kwargs)
        return df

    def _load_cached_data(self) -> Optional[pd.DataFrame]:
        """尝试加载缓存数据"""
        data_dir = Path(__file__).parent.parent / "data_storage"

        # 优先尝试parquet
        parquet_path = data_dir / f"{self.stock_code}_processed.parquet"
        if parquet_path.exists():
            return pd.read_parquet(parquet_path)

        # 其次尝试csv
        csv_path = data_dir / f"{self.stock_code}_processed.csv"
        if csv_path.exists():
            return self.load_from_csv(csv_path)

        return None

    def _save_cache(self, df: pd.DataFrame):
        """保存缓存数据"""
        data_dir = Path(__file__).parent.parent / "data_storage"
        data_dir.mkdir(exist_ok=True)

        df.to_parquet(data_dir / f"{self.stock_code}_processed.parquet")
        df.to_csv(data_dir / f"{self.stock_code}_processed.csv")

    def load_from_csv(self, file_path: Union[str, Path] = None) -> pd.DataFrame:
        """
        从CSV文件加载数据

        CSV格式要求：
            - 必须包含列：datetime, open, high, low, close, volume
            - datetime作为索引
        """
        if file_path is None:
            data_dir = Path(__file__).parent.parent / "data_storage"
            file_path = data_dir / f"{self.stock_code}_raw.csv"

        df = pd.read_csv(file_path)

        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)
        elif 'date' in df.columns:
            df['datetime'] = pd.to_datetime(df['date'])
            df.set_index('datetime', inplace=True)

        df = df.sort_index()
        return df

    def load_from_parquet(self, file_path: Union[str, Path] = None) -> pd.DataFrame:
        """从Parquet文件加载数据"""
        if file_path is None:
            data_dir = Path(__file__).parent.parent / "data_storage"
            file_path = data_dir / f"{self.stock_code}_raw.parquet"

        df = pd.read_parquet(file_path)
        df = df.sort_index()
        return df

    def load_sample_data(self, days: int = 365) -> pd.DataFrame:
        """
        生成模拟数据（用于演示）

        在无法获取真实数据时使用，生成模拟的OHLCV数据
        """
        np.random.seed(settings.RANDOM_STATE)

        # 创建时间索引（5分钟间隔）
        periods = days * 48  # 每天48个5分钟K线（4小时交易时间）
        start_date = pd.to_datetime(settings.START_DATE)
        timestamps = pd.date_range(start=start_date, periods=periods, freq='5min')

        # 生成模拟价格数据
        base_price = 50.0  # 中芯国际基准价格
        returns = np.random.normal(0.001, 0.02, periods)
        price_path = base_price * (1 + returns).cumprod()

        # 生成OHLCV
        open_prices = price_path
        high_prices = price_path * (1 + np.random.uniform(0, 0.01, periods))
        low_prices = price_path * (1 - np.random.uniform(0, 0.01, periods))
        close_prices = low_prices + (high_prices - low_prices) * np.random.uniform(0, 1, periods)
        volumes = np.random.randint(100000, 1000000, periods)

        df = pd.DataFrame({
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volumes
        }, index=timestamps)

        # 确保价格逻辑正确
        df['high'] = df[['open', 'close', 'high']].max(axis=1)
        df['low'] = df[['open', 'close', 'low']].min(axis=1)

        return df

    def save_data(self, df: pd.DataFrame, file_path: Union[str, Path]):
        """保存数据到Parquet文件"""
        df.to_parquet(file_path)

    def validate_data(self, df: pd.DataFrame) -> bool:
        """
        验证数据完整性
        """
        required_columns = ['open', 'high', 'low', 'close', 'volume']

        # 检查必需列
        for col in required_columns:
            if col not in df.columns:
                print(f"缺少必需列: {col}")
                return False

        # 检查缺失值
        if df[required_columns].isnull().any().any():
            print("数据存在缺失值")

        # 检查价格逻辑
        invalid_rows = (
            (df['high'] < df['low']) |
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close'])
        )
        if invalid_rows.any():
            print(f"发现{invalid_rows.sum()}行价格逻辑错误")

        return True

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        数据预处理
        """
        df = df.copy()

        # 确保索引是datetime且排序
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df = df.sort_index()

        # 处理缺失值（前向填充）
        df = df.fillna(method='ffill').fillna(method='bfill')

        # 验证数据
        self.validate_data(df)

        return df


if __name__ == "__main__":
    print("测试数据加载器...")
    loader = DataLoader()
    df = loader.load_stock_data(source="sample")
    print(f"\n数据加载完成，共{len(df)}条记录")
    print(df.head())
