import pandas as pd
import numpy as np

class OrderFlowFeatures:
    def __init__(self):
        pass

    def calculate_oi_velocity(self, df: pd.DataFrame, time_col='timestamp', oi_col='oi') -> pd.Series:
        """
        Calculates the rate of change of OI per minute (or per snapshot).
        """
        # Assumes df is sorted by time and grouped by strike/option_type if necessary
        # Usually applied to aggregate OI or specific strikes
        df = df.sort_values(by=time_col)
        df['time_diff'] = df[time_col].diff().dt.total_seconds() / 60.0
        return df[oi_col].diff() / df['time_diff']

    def calculate_strike_concentration(self, df: pd.DataFrame) -> float:
        """
        Calculates HHI (Herfindahl-Hirschman Index) of OI distribution across strikes.
        Higher HHI means OI is concentrated in few strikes (potential pin risk).
        """
        total_oi = df['oi'].sum()
        if total_oi == 0:
            return 0.0
        
        shares = df.groupby('strike')['oi'].sum() / total_oi
        hhi = (shares ** 2).sum()
        return hhi

    def calculate_atm_imbalance(self, df: pd.DataFrame, atm_strike: float) -> float:
        """
        Calculates (Call OI - Put OI) / (Call OI + Put OI) at the ATM strike.
        """
        atm_df = df[df['strike'] == atm_strike]
        if atm_df.empty:
            return 0.0
        
        call_oi = atm_df[atm_df['option_type'] == 'CE']['oi'].sum()
        put_oi = atm_df[atm_df['option_type'] == 'PE']['oi'].sum()
        
        total = call_oi + put_oi
        if total == 0:
            return 0.0
            
        return (call_oi - put_oi) / total
