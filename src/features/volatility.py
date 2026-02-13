import pandas as pd
import numpy as np

class VolatilityFeatures:
    def __init__(self, window_days=60):
        self.window_days = window_days

    def calculate_iv_percentile(self, df: pd.DataFrame, iv_col='iv') -> pd.Series:
        """
        Calculates the IV Percentile over a rolling window.
        IV Percentile = (Current IV < Past IVs) / Total Past IVs
        """
        # Assuming 1 data point per day for long-term stats, or we resample.
        # If intraday, we might look at the last N days relative to now.
        # Here we implement a simple rolling rank pct.
        return df[iv_col].rolling(window=self.window_days).rank(pct=True) * 100

    def calculate_iv_zscore(self, df: pd.DataFrame, iv_col='iv') -> pd.Series:
        """
        Calculates IV Z-Score based on rolling mean and std dev.
        """
        roll_mean = df[iv_col].rolling(window=self.window_days).mean()
        roll_std = df[iv_col].rolling(window=self.window_days).std()
        return (df[iv_col] - roll_mean) / (roll_std + 1e-9)

    def calculate_iv_expansion_velocity(self, df: pd.DataFrame, iv_col='iv', period=5) -> pd.Series:
        """
        Calculates the rate of change of IV over a short period.
        """
        return df[iv_col].diff(period)

    def calculate_atr_percentile(self, df: pd.DataFrame, period=14, lookback=60) -> pd.Series:
        """
        Calculates Average True Range (ATR) and its percentile rank.
        Requires 'high', 'low', 'close' columns.
        """
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        # ATR Percentile
        return atr.rolling(window=lookback).rank(pct=True) * 100
