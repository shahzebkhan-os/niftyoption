import pandas as pd
import numpy as np

def compute_otm_trap(df: pd.DataFrame, threshold=2.0):
    """
    Detects OTM Trap scenarios where high volume and high IV z-score occur 
    without corresponding price movement, signaling potential reversals or squeezes.
    """
    # Requires 'volume', 'iv_zscore', and 'underlying_price'
    
    # 1. Volume Spike (Relative to 20-period average)
    df['otm_volume_spike'] = df.groupby(['strike', 'option_type'])['volume'].apply(
        lambda x: x > x.rolling(20).mean() * threshold
    ).reset_index(level=[0,1], drop=True)

    # 2. Trap Score Calculation
    # +1 point for volume spike
    # +1 point if IV is expanded (Z-Score > 1)
    # +1 point if Price is congested (abs change < 0.1%)
    
    price_change = df.groupby(['strike', 'option_type'])['underlying_price'].pct_change().abs()
    
    df['trap_score'] = (
        df['otm_volume_spike'].astype(int) +
        (df['iv_zscore'] > 1.0).astype(int) +
        (price_change < 0.001).astype(int)
    )

    return df
