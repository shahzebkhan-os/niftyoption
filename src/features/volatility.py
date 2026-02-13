import pandas as pd
import numpy as np
from scipy.stats import norm

def black_scholes_greeks(S, K, T, r, sigma, option_type='CE'):
    """
    Computes Black-Scholes price and greeks (Delta, Gamma, Theta, Vega).
    T is in years.
    """
    # Use small epsilon to avoid division by zero
    eps = 1e-7
    T = max(T, eps)
    sigma = max(sigma, eps)
    
    try:
        d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == 'CE':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
            delta = norm.cdf(d1)
            theta = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) 
                     - r * K * np.exp(-r * T) * norm.cdf(d2))
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            delta = norm.cdf(d1) - 1
            theta = (- (S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) 
                     + r * K * np.exp(-r * T) * norm.cdf(-d2))
            
        gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        vega = S * norm.pdf(d1) * np.sqrt(T)
        
        return price, delta, gamma, theta, vega
    except:
        return 0.0, 0.0, 0.0, 0.0, 0.0

def find_iv(market_price, S, K, T, r, option_type='CE', precision=0.0001, max_iter=100):
    """
    Finds Implied Volatility using Newton-Raphson.
    """
    if market_price <= 0:
        return 0.0001
        
    sigma = 0.5  # Initial guess
    for i in range(max_iter):
        price, _, _, _, vega = black_scholes_greeks(S, K, T, r, sigma, option_type)
        diff = market_price - price
        if abs(diff) < precision:
            return float(sigma)
        if vega > 1e-4:
            sigma = sigma + diff / vega 
            # Clamp sigma to reasonable bounds
            sigma = max(0.0001, min(sigma, 5.0))
        else:
            break
            
    # Binary search fallback
    low, high = 0.0001, 5.0
    for _ in range(20):
        mid = (low + high) / 2
        price, _, _, _, _ = black_scholes_greeks(S, K, T, r, mid, option_type)
        if price < market_price:
            low = mid
        else:
            high = mid
    return float((low + high) / 2)

def enrich_with_greeks(df: pd.DataFrame, r=0.07):
    """
    Enriches dataframe with IV and Greeks.
    Expects columns: ltp, strike, underlying_price, timestamp, expiry, option_type
    """
    if df.empty:
        return df

    # Prepare time to expiry in years
    # Ensure columns are datetime objects
    df['expiry'] = pd.to_datetime(df['expiry'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Drop existing Greeks to avoid duplicate columns
    cols_to_drop = ['iv', 'delta', 'gamma', 'theta', 'vega', 'T']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    days_to_expiry = (df['expiry'] - df['timestamp']).dt.days
    df['T'] = days_to_expiry / 365.0
    df['T'] = df['T'].clip(lower=1/365.0) # Avoid T=0

    # Calculate IV only for CE/PE
    def row_iv(row):
        if row['option_type'] not in ['CE', 'PE']:
            return 0.0
        try:
            return find_iv(row['ltp'], row['underlying_price'], row['strike'], row['T'], r, row['option_type'])
        except:
            return 0.2

    df['iv'] = df.apply(row_iv, axis=1)

    # Calculate Greeks only for CE/PE
    def row_greeks(row):
        if row['option_type'] not in ['CE', 'PE']:
            return pd.Series({'delta': 0.0, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0})
        _, d, g, t, v = black_scholes_greeks(row['underlying_price'], row['strike'], row['T'], r, row['iv'], row['option_type'])
        return pd.Series({'delta': d, 'gamma': g, 'theta': t, 'vega': v})

    greeks = df.apply(row_greeks, axis=1)
    df = pd.concat([df, greeks], axis=1)
    
    return df

def compute_iv_percentile(df: pd.DataFrame, window=60):
    """
    Computes IV Percentile, Z-Score, and basic stats over a rolling window.
    Ensures non-leakage by using only past data.
    """
    if 'iv' not in df.columns:
        return df
        
    # 1. Rolling Mean & Std
    df['iv_mean'] = df['iv'].rolling(window, min_periods=min(10, len(df))).mean()
    df['iv_std'] = df['iv'].rolling(window, min_periods=min(10, len(df))).std()
    
    # 2. IV Z-Score
    df['iv_zscore'] = np.where(df['iv_std'] > 0, (df['iv'] - df['iv_mean']) / df['iv_std'], 0)

    # 3. IV Percentile
    df['iv_percentile'] = (
        df['iv']
        .rolling(window, min_periods=min(10, len(df)))
        .apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0)
    ) * 100

    return df
