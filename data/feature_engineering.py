"""
特征工程模块
自动化特征生成和时序特征提取
"""
import pandas as pd
import numpy as np
from typing import List, Optional
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import settings


class FeatureEngineer:
    """特征工程器"""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.close = df['close']

    def build_all_features(self) -> pd.DataFrame:
        """构建所有特征"""
        df = self.df.copy()

        # 价格变化特征
        df = self.add_price_features(df)

        # 滚动统计特征
        df = self.add_rolling_statistics(df)

        # 时间特征
        df = self.add_time_features(df)

        # 技术指标衍生特征
        df = self.add_indicator_features(df)

        return df

    def add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加价格变化特征"""
        # 简单收益率
        df['Return_1'] = self.close.pct_change(1)
        df['Return_5'] = self.close.pct_change(5)
        df['Return_10'] = self.close.pct_change(10)

        # 对数收益率
        df['LogReturn_1'] = np.log(self.close / self.close.shift(1))
        df['LogReturn_5'] = np.log(self.close / self.close.shift(5))

        # 价格相对位置
        df['Close_to_Open'] = (df['close'] - df['open']) / df['open']
        df['Close_to_High'] = (df['close'] - df['high']) / df['high']
        df['Close_to_Low'] = (df['close'] - df['low']) / df['low']
        df['High_to_Low'] = (df['high'] - df['low']) / df['low']

        # 相对于均线的位置
        for period in settings.MA_PERIODS:
            ma_col = f'MA{period}'
            if ma_col in df.columns:
                df[f'Close_to_{ma_col}'] = (self.close - df[ma_col]) / df[ma_col]

        return df

    def add_rolling_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加滚动统计特征"""
        windows = settings.ROLLING_WINDOWS
        features = ['close', 'volume']

        for window in windows:
            for feature in features:
                if feature in df.columns:
                    # 均值
                    df[f'{feature}_Mean_{window}'] = df[feature].rolling(window=window).mean()
                    # 标准差
                    df[f'{feature}_Std_{window}'] = df[feature].rolling(window=window).std()
                    # 最大值
                    df[f'{feature}_Max_{window}'] = df[feature].rolling(window=window).max()
                    # 最小值
                    df[f'{feature}_Min_{window}'] = df[feature].rolling(window=window).min()
                    # 偏度
                    df[f'{feature}_Skew_{window}'] = df[feature].rolling(window=window).skew()
                    # 峰度
                    df[f'{feature}_Kurt_{window}'] = df[feature].rolling(window=window).kurt()

        # 滚动相关性
        if 'volume' in df.columns:
            for window in windows:
                df[f'Close_Volume_Corr_{window}'] = (
                    df['close'].rolling(window=window).corr(df['volume'])
                )

        return df

    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加时间特征"""
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # 小时（0-23）
        df['Hour'] = df.index.hour
        # 分钟（0-59）
        df['Minute'] = df.index.minute
        # 星期几（0-6）
        df['DayOfWeek'] = df.index.dayofweek
        # 一个月中的第几天（1-31）
        df['DayOfMonth'] = df.index.day
        # 月份（1-12）
        df['Month'] = df.index.month

        # 交易日时段特征
        df['IsMorning'] = (df.index.hour < 12).astype(int)
        df['IsAfternoon'] = (df.index.hour >= 13).astype(int)

        return df

    def add_indicator_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加技术指标衍生特征"""
        # MACD变化率
        if 'MACD' in df.columns:
            df['MACD_Change'] = df['MACD'].diff()

        # RSI超买超卖区域
        if 'RSI' in df.columns:
            df['RSI_Overbought'] = (df['RSI'] > 70).astype(int)
            df['RSI_Oversold'] = (df['RSI'] < 30).astype(int)

        # KDJ交叉信号
        if 'K' in df.columns and 'D' in df.columns:
            df['KDJ_Cross_Up'] = ((df['K'] > df['D']) & (df['K'].shift(1) <= df['D'].shift(1))).astype(int)
            df['KDJ_Cross_Down'] = ((df['K'] < df['D']) & (df['K'].shift(1) >= df['D'].shift(1))).astype(int)

        # BOLL位置
        if all(col in df.columns for col in ['BOLL_Upper', 'BOLL_Lower', 'BOLL_Mid']):
            df['BOLL_Position'] = (df['close'] - df['BOLL_Lower']) / (df['BOLL_Upper'] - df['BOLL_Lower'] + 1e-8)

        # 成交量变化
        if 'volume' in df.columns:
            df['Volume_Change'] = df['volume'].pct_change()
            df['Volume_Ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()

        return df

    def create_target(self, df: pd.DataFrame, periods: int = None) -> pd.DataFrame:
        """
        创建目标变量

        参数:
            periods: 预测未来N期的收益
        """
        if periods is None:
            periods = settings.TARGET_PERIOD

        # 未来收益率
        df['Future_Return'] = df['close'].pct_change(periods).shift(-periods)

        # 分类目标：涨跌（1涨，0跌）
        df['Target_Class'] = (df['Future_Return'] > 0).astype(int)

        # 回归目标：未来收益率
        df['Target_Reg'] = df['Future_Return']

        return df
