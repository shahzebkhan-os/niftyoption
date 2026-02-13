import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class RegimeClassifier:
    def __init__(self,
                 ema_fast=9,
                 ema_slow=21,
                 atr_window=14,
                 iv_threshold=0.7):
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.atr_window = atr_window
        self.iv_threshold = iv_threshold

    def detect_regime(self, df: pd.DataFrame):
        """
        Classifies the current market regime based on Trend Strength, 
        ATR Percentile, and IV Percentile.
        """
        if df.empty or len(df) < max(self.ema_slow, self.atr_window):
            return "RANGE_BOUND"

        # Always operate on past data only (df should already be filtered/lagged if called from pipeline)
        latest = df.iloc[-1]

        # Extract metrics
        trend_strength = latest.get('ema_fast', 0) - latest.get('ema_slow', 0)
        atr_pct = latest.get('atr_percentile', 0.5)
        iv_high = latest.get('iv_percentile', 0) > (self.iv_threshold * 100)

        # 1. TRENDING UP
        if trend_strength > 0 and atr_pct > 0.6:
            return "TRENDING_UP"

        # 2. TRENDING DOWN
        elif trend_strength < 0 and atr_pct > 0.6:
            return "TRENDING_DOWN"

        # 3. HIGH VOL EXPANSION
        elif iv_high and atr_pct > 0.7:
            return "VOLATILITY_EXPANSION"

        # 4. RANGE BOUND (Default)
        else:
            return "RANGE_BOUND"

    def regime_confidence(self, df: pd.DataFrame):
        """
        Calculates a confidence score (0.0 to 1.0) for the current regime classification.
        """
        if df.empty:
            return 0.0

        latest = df.iloc[-1]

        # NormalizedTrend (Relative to price)
        price = latest.get('underlying_price', latest.get('price'))
        if price is None or price == 0:
            return 0.5 # Default middle probability
            
        trend_score = abs(latest.get('ema_fast', 0) - latest.get('ema_slow', 0)) / price
        # Scale trend score (e.g., 0.5% diff is high confidence)
        trend_score = np.clip(trend_score / 0.005, 0, 1)

        vol_score = latest.get('atr_percentile', 0)
        iv_score = latest.get('iv_percentile', 0) / 100.0

        confidence = np.clip(
            0.4 * trend_score +
            0.3 * vol_score +
            0.3 * iv_score,
            0, 1
        )

        return confidence

    def regime_stability(self, df: pd.DataFrame, lookback=20):
        """
        Detects frequent regime flips. Higher score means more stability.
        1.0 = No transitions in lookback.
        0.0 = transitioning every bar.
        """
        if 'regime' not in df.columns or len(df) < lookback:
            return 1.0

        regimes = df['regime'].tail(lookback)
        transitions = (regimes != regimes.shift(1)).sum()
        
        # Subtract the first NaN shift
        if transitions > 0: transitions -= 1

        stability = 1 - (transitions / lookback)
        return np.clip(stability, 0, 1)
