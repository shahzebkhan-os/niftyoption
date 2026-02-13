import pandas as pd
import numpy as np
import logging
from .volatility import compute_iv_percentile
from .dealer_positioning import compute_gex
from .orderflow import compute_oi_velocity, compute_divergence
from .trap_detection import compute_otm_trap
from src.strategy.regime_classifier import RegimeClassifier

logger = logging.getLogger(__name__)

def build_features(df: pd.DataFrame, lag_safety=True):
    """
    Orchestrates the full multi-step feature computation process.
    Unifies Volatility, GEX, Order Flow, Trap Detection, and Regime Analysis.
    """
    if df.empty:
        return pd.DataFrame()

    logger.info(f"Building features for {len(df)} records...")

    # Initialize Regime Classifier
    regime_clf = RegimeClassifier()

    # 1. Preprocessing & Technicals (Price Level)
    df = df.sort_values(['timestamp', 'strike', 'option_type'])
    
    # Calculate underlying OHLC for ATR and EMAs
    underlying = df.groupby('timestamp')['underlying_price'].first().to_frame()
    underlying['ema_fast'] = underlying['underlying_price'].ewm(span=regime_clf.ema_fast, adjust=False).mean()
    underlying['ema_slow'] = underlying['underlying_price'].ewm(span=regime_clf.ema_slow, adjust=False).mean()
    
    # ATR approximation (using 1min absolute changes)
    underlying['tr'] = underlying['underlying_price'].diff().abs()
    underlying['atr'] = underlying['tr'].rolling(window=regime_clf.atr_window).mean()
    underlying['atr_percentile'] = underlying['atr'].rolling(50).rank(pct=True)
    
    # Merge technicals back
    df = df.merge(underlying[['ema_fast', 'ema_slow', 'atr', 'atr_percentile']], on='timestamp', how='left')

    # 2. Sequential Option Feature Computation (Contract Level)
    df = compute_iv_percentile(df)
    df = compute_oi_velocity(df)
    df = compute_divergence(df)
    df = compute_otm_trap(df)

    # 3. Aggregate Feature Computation (Timestamp Level)
    net_gex = compute_gex(df)
    df = df.merge(net_gex, on='timestamp', how='left')

    # 4. Regime Classification (Timestamp Level)
    # We apply regime detection per unique timestamp
    regime_results = []
    
    # To avoid repeated calculations, we'll process unique timestamps in order
    ts_groups = df.groupby('timestamp')
    rolling_df = pd.DataFrame()
    
    for ts, group in ts_groups:
        # Build a rolling view for stability calculation
        # In a production pipeline, this might be optimized
        rolling_df = pd.concat([rolling_df, group.head(1)])
        
        regime = regime_clf.detect_regime(rolling_df)
        conf = regime_clf.regime_confidence(rolling_df)
        
        # Add regime to the rolling df for stability check
        rolling_df.loc[rolling_df['timestamp'] == ts, 'regime'] = regime
        stability = regime_clf.regime_stability(rolling_df)
        
        regime_results.append({
            'timestamp': ts,
            'regime': regime,
            'regime_confidence': conf,
            'regime_stability': stability
        })
        
    regime_df = pd.DataFrame(regime_results)
    df = df.merge(regime_df, on='timestamp', how='left')

    # Convert Categorical Regime to Numerical for ML if needed
    regime_map = {"TRENDING_UP": 1, "TRENDING_DOWN": -1, "VOLATILITY_EXPANSION": 2, "RANGE_BOUND": 0}
    df['regime_encoded'] = df['regime'].map(regime_map)

    # 5. Clean Data
    df = df.dropna()

    # 6. MANDATORY LAG SAFETY (Shield against leakage)
    if lag_safety and not df.empty:
        # We shift features AND labels (regime) forward by 1 observation 
        # to ensure that at time T, the model selection and input features 
        # only use information that was available at T-1.
        shift_cols = [
            'iv_percentile', 'iv_zscore', 'oi_velocity', 'oi_acceleration', 
            'divergence', 'trap_score', 'net_gex', 
            'regime', 'regime_encoded', 'regime_confidence', 'regime_stability'
        ]
        df[shift_cols] = df.groupby(['strike', 'option_type'])[shift_cols].shift(1)
        df = df.dropna(subset=shift_cols)

    logger.info(f"Feature engineering complete. Final shape: {df.shape}")
    return df
