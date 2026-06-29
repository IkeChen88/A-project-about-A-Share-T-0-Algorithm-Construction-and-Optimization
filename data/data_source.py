"""
真实数据源获取模块
支持从akshare或tushare获取A股历史数据
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
import sys
from datetime import datetime, timedelta
sys.path.append(str(Path(__file__).parent.parent))
from config import settings

# 尝试导入数据源库
try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
except ImportError:
    AKSHARE_AVAILABLE = False

try:
    import tushare as ts
    TUSHARE_AVAILABLE = True
except ImportError:
    TUSHARE_AVAILABLE = False


class StockDataSource:
    """股票数据源"""

    def __init__(self):
        self.stock_code = settings.STOCK_CODE
        self.timeframe = settings.TIMEFRAME

    def get_stock_data_akshare(
        self,
        symbol: str = None,
        period: str = None,
        start_date: str = None,
        end_date: str = None,
        adjust: str = "qfq"
    ) -> pd.DataFrame:
        """
        使用akshare获取股票数据

        参数:
            symbol: 股票代码（例如：688981）
            period: 周期（1, 5, 15, 30, 60分钟，日线等）
            start_date: 开始日期（YYYYMMDD）
            end_date: 结束日期（YYYYMMDD）
            adjust: 复权方式（qfq: 前复权，hfq: 后复权，none: 不复权）

        返回:
            包含OHLCV数据的DataFrame
        """
        if not AKSHARE_AVAILABLE:
            raise ImportError("akshare未安装，请运行：pip install akshare")

        if symbol is None:
            symbol = self.stock_code

        if period is None:
            period = self._map_timeframe(settings.TIMEFRAME)

        if start_date is None:
            start_date = settings.START_DATE.replace("-", "")

        if end_date is None:
            end_date = settings.END_DATE.replace("-", "")

        print(f"正在使用akshare获取 {symbol} 数据...")
        print(f"周期: {period}, 日期范围: {start_date} - {end_date}")

        try:
            # 获取股票历史数据
            if "分钟" in period or "min" in period.lower():
                # 分钟K线
                if "5" in period:
                    period = "5"
                elif "15" in period:
                    period = "15"
                elif "30" in period:
                    period = "30"
                elif "60" in period or "1h" in period.lower():
                    period = "60"

                df = ak.stock_zh_a_hist_min_em(
                    symbol=symbol,
                    period=period,
                    start_date=start_date,
                    end_date=end_date,
                    adjust=adjust
                )
            else:
                # 日线及以上
                df = ak.stock_zh_a_hist(
                    symbol=symbol,
                    period=period,
                    start_date=start_date,
                    end_date=end_date,
                    adjust=adjust
                )

            # 重命名列
            df = self._rename_columns(df, source="akshare")

            print(f"成功获取 {len(df)} 条数据")
            return df

        except Exception as e:
            print(f"akshare获取数据失败: {e}")
            print("尝试使用备用数据获取方式...")
            return self._get_backup_data(symbol, start_date, end_date)

    def get_stock_data_tushare(
        self,
        token: str = None,
        symbol: str = None,
        freq: str = None,
        start_date: str = None,
        end_date: str = None,
        adj: str = "qfq"
    ) -> pd.DataFrame:
        """
        使用tushare获取股票数据（需要token）

        参数:
            token: tushare token
            其他参数同akshare

        返回:
            包含OHLCV数据的DataFrame
        """
        if not TUSHARE_AVAILABLE:
            raise ImportError("tushare未安装，请运行：pip install tushare")

        if token is not None:
            ts.set_token(token)

        if symbol is None:
            symbol = self.stock_code

        if freq is None:
            freq = self._map_timeframe(settings.TIMEFRAME, source="tushare")

        print(f"正在使用tushare获取 {symbol} 数据...")

        try:
            pro = ts.pro_api()

            # 获取日线数据
            if freq == "D":
                ts_code = self._convert_to_ts_code(symbol)
                df = pro.daily(
                    ts_code=ts_code,
                    start_date=start_date,
                    end_date=end_date,
                    adj=adj
                )
                df = df.sort_values('trade_date')
                df = self._rename_columns(df, source="tushare")
            else:
                # 分钟数据处理较复杂，此处简化
                print("tushare分钟数据需要额外配置，使用akshare代替")
                return self.get_stock_data_akshare(symbol, freq, start_date, end_date)

            print(f"成功获取 {len(df)} 条数据")
            return df

        except Exception as e:
            print(f"tushare获取数据失败: {e}")
            return self._get_backup_data(symbol, start_date, end_date)

    def get_stock_data(
        self,
        source: str = "auto",
        **kwargs
    ) -> pd.DataFrame:
        """
        获取股票数据的统一接口

        参数:
            source: 数据源选择（auto, akshare, tushare, sample）
            **kwargs: 传递给具体数据源的参数

        返回:
            包含OHLCV数据的DataFrame
        """
        if source == "auto":
            if AKSHARE_AVAILABLE:
                return self.get_stock_data_akshare(**kwargs)
            elif TUSHARE_AVAILABLE:
                return self.get_stock_data_tushare(**kwargs)
            else:
                print("未找到可用数据源，使用模拟数据")
                return self._get_backup_data()

        elif source == "akshare":
            return self.get_stock_data_akshare(**kwargs)

        elif source == "tushare":
            return self.get_stock_data_tushare(**kwargs)

        elif source == "sample":
            return self._get_backup_data()

        else:
            raise ValueError(f"未知数据源: {source}")

    def _map_timeframe(self, timeframe: str, source: str = "akshare") -> str:
        """映射时间周期"""
        if source == "akshare":
            mapping = {
                "1min": "1", "5min": "5", "15min": "15", "30min": "30", "60min": "60",
                "1d": "daily", "1w": "weekly", "1M": "monthly"
            }
        elif source == "tushare":
            mapping = {
                "1min": "1min", "5min": "5min", "15min": "15min", "30min": "30min", "60min": "60min",
                "1d": "D", "1w": "W", "1M": "M"
            }
        else:
            return timeframe

        return mapping.get(timeframe, "5")

    def _convert_to_ts_code(self, symbol: str) -> str:
        """转换股票代码格式"""
        if symbol.startswith("6"):
            return f"{symbol}.SH"
        elif symbol.startswith("0") or symbol.startswith("3"):
            return f"{symbol}.SZ"
        return symbol

    def _rename_columns(self, df: pd.DataFrame, source: str) -> pd.DataFrame:
        """重命名列以统一格式"""
        df = df.copy()

        if source == "akshare":
            column_mapping = {
                "日期": "datetime",
                "时间": "datetime",
                "开盘": "open",
                "最高": "high",
                "最低": "low",
                "收盘": "close",
                "成交量": "volume"
            }
        elif source == "tushare":
            column_mapping = {
                "trade_date": "datetime",
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "vol": "volume"
            }
        else:
            return df

        df = df.rename(columns=column_mapping)

        # 设置datetime索引
        if "datetime" in df.columns:
            df["datetime"] = pd.to_datetime(df["datetime"])
            df = df.set_index("datetime")

        # 确保包含必需列
        required_columns = ["open", "high", "low", "close", "volume"]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"缺少必需列: {col}")

        return df.sort_index()

    def _get_backup_data(
        self,
        symbol: str = None,
        start_date: str = None,
        end_date: str = None
    ) -> pd.DataFrame:
        """
        获取备用数据（模拟数据）
        在无法获取真实数据时使用
        """
        print("生成模拟数据作为备用...")

        np.random.seed(settings.RANDOM_STATE)

        # 生成近两年的5分钟数据
        end_dt = datetime.now()
        start_dt = end_dt - timedelta(days=365 * 2)

        # 只保留交易日的交易时间
        dates = pd.date_range(start=start_dt, end=end_dt, freq='5min')
        # 过滤出交易时间（简化处理：周一至周五，9:30-11:30，13:00-15:00）
        mask = ((dates.weekday < 5) &
                (((dates.hour == 9) & (dates.minute >= 30)) |
                 ((dates.hour == 10) | (dates.hour == 11) | (dates.hour == 13)) |
                 ((dates.hour == 14) | (dates.hour == 15) & (dates.minute == 0))))
        dates = dates[mask]

        # 生成模拟价格
        base_price = 50.0
        returns = np.random.normal(0.0005, 0.015, len(dates))
        price_path = base_price * (1 + returns).cumprod()

        open_prices = price_path
        high_prices = price_path * (1 + np.random.uniform(0, 0.01, len(dates)))
        low_prices = price_path * (1 - np.random.uniform(0, 0.01, len(dates)))
        close_prices = low_prices + (high_prices - low_prices) * np.random.uniform(0.3, 0.7, len(dates))
        volumes = np.random.randint(50000, 500000, len(dates))

        df = pd.DataFrame({
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volumes
        }, index=dates)

        df['high'] = df[['open', 'close', 'high']].max(axis=1)
        df['low'] = df[['open', 'close', 'low']].min(axis=1)

        print(f"生成模拟数据: {len(df)} 条记录")
        return df

    def save_data_to_csv(self, df: pd.DataFrame, file_path: str):
        """保存数据到CSV"""
        df.to_csv(file_path, index=True)
        print(f"数据已保存到: {file_path}")

    def save_data_to_parquet(self, df: pd.DataFrame, file_path: str):
        """保存数据到Parquet"""
        df.to_parquet(file_path)
        print(f"数据已保存到: {file_path}")


def download_and_save_stock_data(
    symbol: str = "688981",
    data_dir: str = None,
    source: str = "auto",
    start_date: str = None,
    end_date: str = None
) -> pd.DataFrame:
    """
    下载并保存股票数据的便捷函数

    参数:
        symbol: 股票代码
        data_dir: 保存目录
        source: 数据源
        start_date: 开始日期
        end_date: 结束日期

    返回:
        包含OHLCV数据的DataFrame
    """
    if data_dir is None:
        data_dir = Path(__file__).parent.parent / "data_storage"
    data_dir = Path(data_dir)
    data_dir.mkdir(exist_ok=True)

    source = StockDataSource()
    df = source.get_stock_data(
        source=source,
        symbol=symbol,
        start_date=start_date,
        end_date=end_date
    )

    # 保存数据
    csv_path = data_dir / f"{symbol}_raw.csv"
    parquet_path = data_dir / f"{symbol}_raw.parquet"

    source.save_data_to_csv(df, csv_path)
    source.save_data_to_parquet(df, parquet_path)

    return df


def check_data_source_available() -> dict:
    """检查可用的数据源"""
    return {
        "akshare": AKSHARE_AVAILABLE,
        "tushare": TUSHARE_AVAILABLE
    }


if __name__ == "__main__":
    print("数据源测试...")
    available = check_data_source_available()
    print(f"可用数据源: {available}")

    source = StockDataSource()
    df = source.get_stock_data(source="auto")
    print(f"\n数据预览:")
    print(df.head())
    print(f"\n数据统计:")
    print(df.describe())
