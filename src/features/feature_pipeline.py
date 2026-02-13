import pandas as pd
import numpy as np
import logging
import traceback
from .volatility import compute_iv_percentile, enrich_with_greeks
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

    # Pre-check: Ensure critical columns exist
    required = ['ltp', 'strike', 'underlying_price', 'timestamp', 'expiry', 'option_type']
    missing = [col for col in required if col not in df.columns]
    if missing:
        logger.warning(f"Missing critical columns for feature engineering: {missing}")
        return pd.DataFrame()

    # Sanitization: Force unique columns and unique integer index
    df = df.loc[:, ~df.columns.duplicated()].copy()
    df = df.reset_index(drop=True)

    logger.info(f"Building features for {len(df)} records...")

    # Initialize Regime Classifier
    regime_clf = RegimeClassifier()

    # 1. Preprocessing & Technicals (Price Level)
    # Deduplicate and Reset index to avoid label issues during assignment/merge
    # We keep the first occurrence of any unique contract state
    df = df.drop_duplicates(subset=['timestamp', 'strike', 'option_type', 'symbol'], keep='first')
    df = df.sort_values(['timestamp', 'strike', 'option_type']).reset_index(drop=True)
    
    # Calculate underlying OHLC for ATR and EMAs
    underlying = df.groupby('timestamp')['underlying_price'].first().to_frame()
    underlying['ema_fast'] = underlying['underlying_price'].ewm(span=regime_clf.ema_fast, adjust=False).mean()
    underlying['ema_slow'] = underlying['underlying_price'].ewm(span=regime_clf.ema_slow, adjust=False).mean()
    
    # ATR approximation (using 1min absolute changes)
    underlying['tr'] = underlying['underlying_price'].diff().abs()
    underlying['atr'] = underlying['tr'].rolling(window=regime_clf.atr_window, min_periods=min(5, len(underlying))).mean()
    underlying['atr_percentile'] = underlying['atr'].rolling(50, min_periods=min(10, len(underlying))).rank(pct=True)
    
    # Merge technicals back
    df = df.merge(underlying[['ema_fast', 'ema_slow', 'atr', 'atr_percentile']], on='timestamp', how='left').reset_index(drop=True)

    # 2. Black-Scholes Enrichment (Enrich with IV and Greeks if missing)
    df = enrich_with_greeks(df)

    # 3. Sequential Option Feature Computation (Contract Level)
    df = compute_iv_percentile(df)
    df = compute_oi_velocity(df)
    df = compute_divergence(df)
    df = compute_otm_trap(df)

    # 3. Aggregate Feature Computation (Timestamp Level)
    # Ensure net_gex doesn't collide
    df = df.drop(columns=[c for c in ['net_gex'] if c in df.columns])
    
    net_gex = compute_gex(df)
    net_gex_df = net_gex.to_frame().reset_index()
    df = df.merge(net_gex_df, on='timestamp', how='left').reset_index(drop=True)

    # 4. Regime Classification (Timestamp Level)
    # Ensure regime columns don't collide
    regime_cols = ['regime', 'regime_confidence', 'regime_stability', 'regime_encoded']
    df = df.drop(columns=[c for c in regime_cols if c in df.columns])
    
    regime_results = []
    ts_groups = df.groupby('timestamp')
    rolling_df = pd.DataFrame()
    
    for ts, group in ts_groups:
        rolling_df = pd.concat([rolling_df, group.head(1)])
        
        regime = regime_clf.detect_regime(rolling_df)
        conf = regime_clf.regime_confidence(rolling_df)
        
        # Build stability based on rolling window
        rolling_df.loc[rolling_df['timestamp'] == ts, 'regime'] = regime
        stability = regime_clf.regime_stability(rolling_df)
        
        regime_results.append({
            'timestamp': ts,
            'regime': regime,
            'regime_confidence': conf,
            'regime_stability': stability
        })
        
    regime_df = pd.DataFrame(regime_results)
    df = df.merge(regime_df, on='timestamp', how='left').reset_index(drop=True)

    # Convert Categorical Regime to Numerical for ML if needed
    regime_map = {"TRENDING_UP": 1, "TRENDING_DOWN": -1, "VOLATILITY_EXPANSION": 2, "RANGE_BOUND": 0}
    df['regime_encoded'] = df['regime'].map(regime_map)

    # 5. Clean Data - RELAXED
    df = df.dropna(subset=['underlying_price'])
    
    greeks_cols = ['iv', 'oi', 'oi_velocity', 'oi_acceleration', 'divergence', 'trap_score', 'net_gex', 'iv_percentile', 'iv_zscore']
    for col in greeks_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # 6. MANDATORY LAG SAFETY (Shield against leakage)
    if lag_safety and not df.empty:
        shift_cols = [
            'iv_percentile', 'iv_zscore', 'oi_velocity', 'oi_acceleration', 
            'divergence', 'trap_score', 'net_gex', 
            'regime', 'regime_encoded', 'regime_confidence', 'regime_stability',
            'ema_fast', 'ema_slow', 'atr', 'atr_percentile'
        ]
        # Ensure shift_cols exist in df
        shift_cols = [c for c in shift_cols if c in df.columns]
        df[shift_cols] = df.groupby(['strike', 'option_type'])[shift_cols].shift(1)

    logger.info(f"Feature engineering complete. Final shape: {df.shape}")
    return df
