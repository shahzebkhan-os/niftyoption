import pandas as pd
import numpy as np

class TargetGenerator:
    def __init__(self, horizon_minutes=30, threshold_points=20):
        self.horizon_minutes = horizon_minutes
        self.threshold_points = threshold_points

    def generate_target(self, df: pd.DataFrame, price_col='underlying_price', time_col='timestamp') -> pd.Series:
        """
        Generates binary target: 1 if |Price(t+T) - Price(t)| > X, else 0.
        """
        # Ensure df is sorted by time
        df = df.sort_values(by=time_col)
        
        # Calculate future price (shift backwards)
        # Assumes dataframe has uniform 1-minute intervals or re-sampled
        # If not, need to use time-based indexing. 
        # For simplicity, assuming df is 1-min resampled.
        
        future_price = df[price_col].shift(-self.horizon_minutes)
        
        price_change = (future_price - df[price_col]).abs()
        
        target = (price_change > self.threshold_points).astype(int)
        
        # Mask the last 'horizon' rows as they don't have targets
        target.iloc[-self.horizon_minutes:] = np.nan
        
        return target

    def generate_directional_target(self, df: pd.DataFrame, price_col='underlying_price') -> pd.Series:
        """
        Generates directional target: 1 if Price(t+T) > Price(t) + X (Up),
        -1 if Price(t+T) < Price(t) - X (Down), 0 otherwise.
        """
        future_price = df[price_col].shift(-self.horizon_minutes)
        change = future_price - df[price_col]
        
        conditions = [
            (change > self.threshold_points),
            (change < -self.threshold_points)
        ]
        choices = [1, -1]
        
        target = np.select(conditions, choices, default=0)
        # Handle current NaN at end
        target = pd.Series(target, index=df.index)
        target.iloc[-self.horizon_minutes:] = np.nan
        return target
