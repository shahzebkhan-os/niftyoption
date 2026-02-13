import pandas as pd
import numpy as np

def compute_iv_percentile(df: pd.DataFrame, window=60):
    """
    Computes IV Percentile, Z-Score, and basic stats over a rolling window.
    Ensures non-leakage by using only past data.
    """
    # 1. Rolling Mean & Std
    df['iv_mean'] = df['iv'].rolling(window).mean()
    df['iv_std'] = df['iv'].rolling(window).std()
    
    # 2. IV Z-Score (Distance from mean in standard deviations)
    # Handle division by zero if std is 0
    df['iv_zscore'] = np.where(df['iv_std'] > 0, (df['iv'] - df['iv_mean']) / df['iv_std'], 0)

    # 3. IV Percentile (Rank relative to past N observations)
    # Using window-based rank
    df['iv_percentile'] = (
        df['iv']
        .rolling(window)
        .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    ) * 100

    return df
