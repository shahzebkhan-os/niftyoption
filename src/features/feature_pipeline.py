import pandas as pd
import numpy as np
import logging
from .volatility import compute_iv_percentile
from .dealer_positioning import compute_gex
from .orderflow import compute_oi_velocity, compute_divergence
from .trap_detection import compute_otm_trap

logger = logging.getLogger(__name__)

def build_features(df: pd.DataFrame, lag_safety=True):
    """
    Orchestrates the full multi-step feature computation process.
    Unifies Volatility, GEX, Order Flow, and Trap Detection into a single indexed stream.
    """
    if df.empty:
        return pd.DataFrame()

    logger.info(f"Building features for {len(df)} records...")

    # 1. Basic Preprocessing
    df = df.sort_values(['timestamp', 'strike', 'option_type'])
    
    # 2. Sequential Feature Computation (Contract Level)
    df = compute_iv_percentile(df)
    df = compute_oi_velocity(df)
    df = compute_divergence(df)
    df = compute_otm_trap(df)

    # 3. Aggregate Feature Computation (Timestamp Level)
    # Net GEX is a system-wide metric per snapshot
    net_gex = compute_gex(df)
    
    # 4. Merge Aggregates back to main stream
    df = df.merge(net_gex, on='timestamp', how='left')

    # 5. Clean Data
    df = df.dropna()

    # 6. MANDATORY LAG SAFETY (Shield against leakage)
    if lag_safety and not df.empty:
        # We shift features forward by 1 observation to ensure 
        # that at time T, the model only has information from T-1.
        # This is critical for backtesting integrity.
        feature_cols = [
            'iv_percentile', 'iv_zscore', 'oi_velocity', 'oi_acceleration', 
            'divergence', 'trap_score', 'net_gex'
        ]
        # Only shift the features, keep the timestamp/price alignment for targets later
        df[feature_cols] = df.groupby(['strike', 'option_type'])[feature_cols].shift(1)
        df = df.dropna(subset=feature_cols)

    logger.info(f"Feature engineering complete. Final shape: {df.shape}")
    return df
