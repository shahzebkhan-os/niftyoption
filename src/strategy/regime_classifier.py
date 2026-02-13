import pandas as pd
import numpy as np

class RegimeClassifier:
    def __init__(self):
        pass

    def detect_regime(self, df: pd.DataFrame, volatility_features: dict = None) -> str:
        """
        Classifies the current market regime.
        df should have columns: 'close', 'ema_9', 'ema_21', 'ema_50' (or similar indicators)
        volatility_features: dict containing 'iv_percentile', 'iv_velocity'
        """
        if df is None or df.empty or 'close' not in df.columns:
            return "RANGE"

        # Ensure no None values in comparison columns
        required_cols = ['close', 'ema_9', 'ema_21', 'ema_50']
        for col in required_cols:
            if col not in df.columns or df[col].iloc[-1] is None or pd.isna(df[col].iloc[-1]):
                return "RANGE"

        # 1. Trend Filter
        ema_aligned_up = (df['close'].iloc[-1] > df['ema_9'].iloc[-1] > df['ema_21'].iloc[-1] > df['ema_50'].iloc[-1])
        ema_aligned_down = (df['close'].iloc[-1] < df['ema_9'].iloc[-1] < df['ema_21'].iloc[-1] < df['ema_50'].iloc[-1])
        
        # 2. Volatility Filter
        vol_feats = volatility_features or {}
        iv_percentile = vol_feats.get('iv_percentile', 50)
        iv_expanding = vol_feats.get('iv_velocity', 0) > 0.5
        
        if iv_percentile > 80 and iv_expanding:
            return "HIGH_IV_EXPANSION"
        
        if ema_aligned_up:
            return "TRENDING_UP"
        
        if ema_aligned_down:
            return "TRENDING_DOWN"
            
        # 3. Range / Mean Reversion
        # If ADX < 20 (not implemented here, using proxy)
        # or price is oscillating around EMAs
        return "RANGE"

    def get_regime_weights(self, regime: str) -> dict:
        """
        Returns strategy mixing weights based on regime.
        """
        if regime == "TRENDING_UP":
            return {'trend_following': 0.8, 'mean_reversion': 0.2}
        elif regime == "RANGE":
            return {'trend_following': 0.2, 'mean_reversion': 0.8}
        elif regime == "HIGH_IV_EXPANSION":
             # Safe mode or short vega?
            return {'trend_following': 0.0, 'mean_reversion': 0.0, 'cash': 1.0}
        
        return {'trend_following': 0.5, 'mean_reversion': 0.5}
