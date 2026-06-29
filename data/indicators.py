"""
技术指标计算模块
实现各类技术分析指标
"""
import pandas as pd
import numpy as np
from typing import List
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from config import settings


class TechnicalIndicators:
    """技术指标计算器"""

    def __init__(self, df: pd.DataFrame):
        """
        参数:
            df: 包含OHLCV数据的DataFrame
        """
        self.df = df.copy()
        self.close = df['close']
        self.high = df['high']
        self.low = df['low']
        self.volume = df['volume']

    def add_all_indicators(self) -> pd.DataFrame:
        """添加所有技术指标"""
        df = self.df.copy()

        # 趋势类指标
        df = self.add_ma(df)
        df = self.add_ema(df)
        df = self.add_macd(df)
        df = self.add_boll(df)

        # 动量类指标
        df = self.add_rsi(df)
        df = self.add_kdj(df)
        df = self.add_cci(df)
        df = self.add_roc(df)

        # 成交量类指标
        df = self.add_obv(df)
        df = self.add_vwap(df)

        # 波动率类指标
        df = self.add_atr(df)
        df = self.add_volatility(df)

        return df

    # ========== 趋势类指标 ==========

    def add_ma(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加移动平均线"""
        for period in settings.MA_PERIODS:
            df[f'MA{period}'] = self.close.rolling(window=period).mean()
        return df

    def add_ema(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加指数移动平均线"""
        for period in settings.MA_PERIODS:
            df[f'EMA{period}'] = self.close.ewm(span=period, adjust=False).mean()
        return df

    def add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加MACD指标"""
        ema_fast = self.close.ewm(span=settings.MACD_FAST, adjust=False).mean()
        ema_slow = self.close.ewm(span=settings.MACD_SLOW, adjust=False).mean()
        df['MACD'] = ema_fast - ema_slow
        df['MACD_Signal'] = df['MACD'].ewm(span=settings.MACD_SIGNAL, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
        return df

    def add_boll(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加布林带"""
        sma = self.close.rolling(window=settings.BOLL_PERIOD).mean()
        std = self.close.rolling(window=settings.BOLL_PERIOD).std()
        df['BOLL_Mid'] = sma
        df['BOLL_Upper'] = sma + (std * 2)
        df['BOLL_Lower'] = sma - (std * 2)
        df['BOLL_Width'] = (df['BOLL_Upper'] - df['BOLL_Lower']) / df['BOLL_Mid']
        return df

    # ========== 动量类指标 ==========

    def add_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加RSI指标"""
        delta = self.close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=settings.RSI_PERIOD).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=settings.RSI_PERIOD).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        return df

    def add_kdj(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加KDJ指标"""
        low_min = self.low.rolling(window=settings.KDJ_N).min()
        high_max = self.high.rolling(window=settings.KDJ_N).max()
        rsv = (self.close - low_min) / (high_max - low_min) * 100
        df['K'] = rsv.ewm(com=settings.KDJ_M1 - 1, adjust=False).mean()
        df['D'] = df['K'].ewm(com=settings.KDJ_M2 - 1, adjust=False).mean()
        df['J'] = 3 * df['K'] - 2 * df['D']
        return df

    def add_cci(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加CCI指标"""
        tp = (self.high + self.low + self.close) / 3
        ma_tp = tp.rolling(window=20).mean()
        mad = tp.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean())
        df['CCI'] = (tp - ma_tp) / (0.015 * mad)
        return df

    def add_roc(self, df: pd.DataFrame, period: int = 10) -> pd.DataFrame:
        """添加ROC指标"""
        df['ROC'] = ((self.close - self.close.shift(period)) / self.close.shift(period)) * 100
        return df

    # ========== 成交量类指标 ==========

    def add_obv(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加OBV指标"""
        obv = (np.sign(self.close.diff()) * self.volume).fillna(0).cumsum()
        df['OBV'] = obv
        df['OBV_MA5'] = obv.rolling(window=5).mean()
        return df

    def add_vwap(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加VWAP指标"""
        typical_price = (self.high + self.low + self.close) / 3
        cum_tp_vol = (typical_price * self.volume).cumsum()
        cum_vol = self.volume.cumsum()
        df['VWAP'] = cum_tp_vol / cum_vol
        return df

    # ========== 波动率类指标 ==========

    def add_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加ATR指标"""
        high_low = self.high - self.low
        high_close = np.abs(self.high - self.close.shift())
        low_close = np.abs(self.low - self.close.shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['TR'] = tr
        df['ATR'] = tr.rolling(window=settings.ATR_PERIOD).mean()
        return df

    def add_volatility(self, df: pd.DataFrame, periods: List[int] = None) -> pd.DataFrame:
        """添加波动率指标"""
        if periods is None:
            periods = [5, 10, 20]

        returns = np.log(self.close / self.close.shift(1))

        for period in periods:
            df[f'Volatility_{period}'] = returns.rolling(window=period).std() * np.sqrt(252 * 48)  # 年化

        return df
