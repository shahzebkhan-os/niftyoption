import pandas as pd
import numpy as np

class TargetGenerator:
    def __init__(self, horizon_minutes=30, threshold_points=20):
        self.horizon_minutes = horizon_minutes
        self.threshold_points = threshold_points

    def generate_target(self, df: pd.DataFrame, price_col='underlying_price', time_col='timestamp') -> pd.Series:
        """
        Generates binary target: 1 if |Price(t+T) - Price(t)| > X, else 0.
        Uses time-based shifting to handle multiple rows per timestamp.
        """
        if df.empty:
            return pd.Series(dtype=float)

        # 1. Get unique price-time series
        ts_price = df.groupby(time_col)[price_col].first().sort_index().to_frame()
        
        # 2. Use a rounded index for mapping to ensure 1-min alignment
        # We round original timestamps to the nearest minute for the lookup
        ts_price['lookup_time'] = ts_price.index.round('1min')
        
        # 3. Create a continuous 1-min range for shifting
        start, end = ts_price['lookup_time'].min(), ts_price['lookup_time'].max()
        all_minutes = pd.date_range(start=start, end=end, freq='1min')
        
        # 4. Map existing prices to the 1-min grid
        grid_price = ts_price.groupby('lookup_time')[price_col].first().reindex(all_minutes).ffill()
        
        # 5. Calculate future price on the grid
        future_grid_price = grid_price.shift(-self.horizon_minutes)
        
        # 6. Map back to original timestamps using lookup_time
        ts_price['future_price'] = ts_price['lookup_time'].map(future_grid_price)
        
        # 7. Calculate target
        ts_price['price_change'] = (ts_price['future_price'] - ts_price[price_col]).abs()
        ts_price['target'] = (ts_price['price_change'] > self.threshold_points).astype(float)
        
        # Mask NaNs (where future_price was NaN)
        ts_price.loc[ts_price['future_price'].isna(), 'target'] = np.nan
        
        # 8. Map back to original dataframe
        return df[time_col].map(ts_price['target'])

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
