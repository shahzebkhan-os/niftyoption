import pandas as pd
import numpy as np

class TrendFeatures:
    def __init__(self):
        pass

    def calculate_ema_alignment(self, df: pd.DataFrame, periods=[9, 21, 50], price_col='close') -> int:
        """
        Returns a score based on EMA alignment.
        1.0 if Price > EMA9 > EMA21 > EMA50 (Strong Uptrend)
        -1.0 if Price < EMA9 < EMA21 < EMA50 (Strong Downtrend)
        0.0 otherwise (Choppy/Transition)
        """
        score = 0
        emas = {}
        for p in periods:
            emas[p] = df[price_col].ewm(span=p, adjust=False).mean()
        
        # Check Uptrend
        if (df[price_col].iloc[-1] > emas[9].iloc[-1] > emas[21].iloc[-1] > emas[50].iloc[-1]):
            return 1.0
        # Check Downtrend
        elif (df[price_col].iloc[-1] < emas[9].iloc[-1] < emas[21].iloc[-1] < emas[50].iloc[-1]):
            return -1.0
            
        return 0.0

    def calculate_vwap_distance(self, df: pd.DataFrame) -> float:
        """
        Calculates normalized distance from VWAP.
        (Price - VWAP) / VWAP
        """
        # VWAP = Cumulative(Price * Volume) / Cumulative(Volume)
        # Resets daily. Assumes df is intraday data for ONE day.
        # If df spans multiple days, need to group by date.
        
        # Simplified for rolling window or assuming single day passed
        cum_pv = (df['close'] * df['volume']).cumsum()
        cum_vol = df['volume'].cumsum()
        vwap = cum_pv / cum_vol
        
        current_price = df['close'].iloc[-1]
        current_vwap = vwap.iloc[-1]
        
        return (current_price - current_vwap) / current_vwap

    def detect_otm_trap(self, option_chain_df: pd.DataFrame, underlying_price: float, threshold_vol: int = 10000) -> float:
        """
        Detects if OTM options are seeing anomalous volume without corresponding price move.
        Returns a probability score (0-1).
        """
        # Filter for Deep OTM (e.g., > 2% away)
        # Check volume z-score or raw threshold
        # If High Volume but Delta is low/unchanged -> Trap?
        # Simplified logic: High Vol on OTM Strikes far from price.
        
        otm_calls = option_chain_df[(option_chain_df['option_type'] == 'CE') & (option_chain_df['strike'] > underlying_price * 1.01)]
        otm_puts = option_chain_df[(option_chain_df['option_type'] == 'PE') & (option_chain_df['strike'] < underlying_price * 0.99)]
        
        anomalous_calls = otm_calls[otm_calls['volume'] > threshold_vol]
        anomalous_puts = otm_puts[otm_puts['volume'] > threshold_vol]
        
        if not anomalous_calls.empty or not anomalous_puts.empty:
            return 0.8 # High probability of trap/positioning
            
        return 0.1
