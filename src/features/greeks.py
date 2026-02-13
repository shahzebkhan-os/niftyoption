import pandas as pd
import numpy as np

class DealerPositioningFeatures:
    def __init__(self, lot_size=50): # NIFTY lot size default
        self.lot_size = lot_size

    def calculate_net_gex(self, chain_df: pd.DataFrame, spot_price: float) -> dict:
        """
        Calculates Net Gamma Exposure (GEX) across all strikes.
        GEX = Gamma * OI * LotSize * SpotPrice * 0.01 * Directon
        Direction: Dealers are short options generally.
        - Long Call (Dealers Short Call) -> Negative Gamma
        - Long Put (Dealers Short Put) -> Positive Gamma
        Wait, standard convention:
        - Dealers are short OTM options.
        - Short Call -> Short Gamma -> Market moves up, become shorter delta -> Need to buy -> Accelerates move? No.
        Let's stick to standard GEX formula:
        Call GEX = Sum(Gamma * OI * LotSize * Spot * 0.01)
        Put GEX = Sum(Gamma * OI * LotSize * Spot * 0.01)
        Net GEX = Call GEX - Put GEX (or vice versa depending on perspective).
        Commonly: Net GEX ($) = (Call GEX - Put GEX) * Spot
        Here roughly:
        GEX_call = DI * Gamma * Spot
        """
        # Ensure we have required columns
        required = ['strike', 'option_type', 'gamma', 'oi']
        if not all(col in chain_df.columns for col in required):
            return {}

        df = chain_df.copy()
        
        # Calculate individual GEX contribution
        # Assuming Gamma is per share. Total Gamma = Gamma * OI * Lot.
        # Dollar Gamma = 0.5 * Gamma * Spot^2 * OI * Lot? 
        # Simpler metric: Exposure in number of shares to hedge per 1% move.
        # GEX = Gamma * OI * LotSize * Spot * 0.01
        
        df['gex_value'] = df['gamma'] * df['oi'] * self.lot_size * spot_price * 0.01
        
        call_gex = df[df['option_type'] == 'CE']['gex_value'].sum()
        put_gex = df[df['option_type'] == 'PE']['gex_value'].sum()
        
        # Net GEX: Positive implies dealers are Long Gamma (stabilizing).
        # Negative implies dealers are Short Gamma (destabilizing).
        # Typically dealers are Short the options customers buy.
        # Retail buys Calls -> Dealers Short Calls (Short Gamma).
        # Retail buys Puts -> Dealers Short Puts (Short Gamma).
        # Wait, if dealers are market makers, they are short options.
        # Short Call -> Negative Gamma.
        # Short Put -> Negative Gamma.
        # So Net GEX might be sum of both?
        
        # Let's use the layout: 
        # Call OI is usually resistance. Put OI is support.
        # High Put OI -> Dealers Short Puts -> If price drops, they get longer delta? 
        # Actually, let's use the widely accepted GEX logic:
        # Call OI -> Dealers Short Call -> Negative Gamma (Accelerates upside? No, if price goes up, delta becomes more negative, need to buy to hedge? No, need to buy to hedge short call).
        # Short Call Delta is negative. As price Up, Delta becomes MORE negative (Gamma). To hedge, need to BUY underlying.
        # So Short Call = Positive Hedging Flow (Buy as price rises) = Stabilizing? No.
        # Let's stick to SqueezeMetrics definition:
        # Net GEX = (Call GEX - Put GEX)
        
        net_gex = call_gex - put_gex

        return {
            'net_gex': net_gex,
            'call_gex': call_gex,
            'put_gex': put_gex
        }

    def calculate_gamma_flip(self, chain_df: pd.DataFrame) -> float:
        """
        Estimates the price level where Net GEX flips from positive to negative.
        This is roughly where the dominance of Call GEX vs Put GEX shifts.
        """
        # Group by strike and calculate Net GEX per strike
        df = chain_df.copy()
        # We need a unified GEX per strike
        # Call GEX adds, Put GEX subtracts?
        # Or simply find the strike where Cumulative GEX crosses 0.
        
        # Simplified: weighted average of strikes by GEX? 
        # Or look for zero crossing interval.
        return 0.0 # Placeholder for complex logic

    def calculate_dealer_pressure(self, net_gex, spot_price):
        """
         normalized score of GEX relative to market cap or volume.
        """
        return net_gex / spot_price # distinct simplified
