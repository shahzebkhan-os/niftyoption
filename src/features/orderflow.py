import pandas as pd
import numpy as np

def compute_oi_velocity(df: pd.DataFrame):
    """
    Calculates the first and second derivatives of Open Interest (OI).
    OI Velocity = 1st derivative (speed of position building).
    OI Acceleration = 2nd derivative (change in building speed).
    """
    # Assuming df is sorted by timestamp and strike/type
    df['oi_velocity'] = df.groupby(['strike', 'option_type'])['oi'].diff()
    df['oi_acceleration'] = df.groupby(['strike', 'option_type'])['oi_velocity'].diff()
    return df

def compute_divergence(df: pd.DataFrame, lookback=10):
    """
    Calculates Hidden Divergence between Price Momentum and OI Momentum.
    Positive divergence = Price up, OI down (potentially weak trend).
    Negative divergence = Price down, OI up (potentially aggressive selling).
    """
    # Requires 'close' (underlying_price) and 'oi'
    df['price_mom'] = df.groupby(['strike', 'option_type'])['underlying_price'].pct_change(lookback, fill_method=None)
    df['oi_mom'] = df.groupby(['strike', 'option_type'])['oi'].pct_change(lookback, fill_method=None)

    df['divergence'] = df['price_mom'] - df['oi_mom']
    return df

def synthetic_bias(call_atm_price, put_atm_price, spot_price):
    """
    Calculates the bias of Synthetic Futures relative to Spot.
    (Call ATM - Put ATM) - Spot.
    A significant difference often signals institutional positioning ahead of a move.
    """
    synthetic = call_atm_price - put_atm_price
    return synthetic - spot_price
