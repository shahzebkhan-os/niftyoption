import pandas as pd
import numpy as np

def compute_gex(df: pd.DataFrame, lot_size=50):
    """
    Calculates Gamma Exposure (GEX) across all strikes and aggregate Net GEX.
    """
    # GEX = Gamma * OI * LotSize * Spot
    # Note: df must have 'gamma', 'oi', 'strike', and 'underlying_price'
    # We use 'underlying_price' as the spot.
    
    df['gex'] = df['gamma'] * df['oi'] * lot_size * df['underlying_price']
    
    # Split GEX (Calls vs Puts for directional bias)
    # Typically: Long Calls = Short Dealer gamma, Long Puts = Short Dealer gamma
    # Net GEX is usually Call GEX - Put GEX (Simplified)
    
    net_gex = df.groupby('timestamp').apply(
        lambda x: (x[x['option_type'] == 'CE']['gex'].sum() - x[x['option_type'] == 'PE']['gex'].sum())
    ).rename("net_gex")

    return net_gex

def gamma_flip_level(df: pd.DataFrame):
    """
    Estimates the price level where Net GEX flips from positive to negative.
    Assumes df is a single snapshot.
    """
    if df.empty:
        return 0.0
        
    df_sorted = df.sort_values('strike')
    
    # Per strike GEX attribution
    strike_gex = df_sorted.groupby('strike').apply(
        lambda x: x[x['option_type'] == 'CE']['gex'].sum() - x[x['option_type'] == 'PE']['gex'].sum()
    ).cumsum()
    
    # Find strike where cumulative GEX is closest to zero
    if strike_gex.empty:
        return 0.0
        
    flip_strike = strike_gex.abs().idxmin()
    return float(flip_strike)
