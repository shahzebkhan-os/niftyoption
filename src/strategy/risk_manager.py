class RiskManager:
    def __init__(self, kelly_fraction=0.25):
        self.kelly_fraction = kelly_fraction

    def calculate_position_size(self, account_balance: float, win_probability: float, risk_reward_ratio: float) -> float:
        """
        Calculates position size using Fractional Kelly Criterion.
        Kelly % = (b * p - q) / b
        where b = risk_reward_ratio (Payoff Ratio)
        p = probability of winning
        q = probability of losing (1 - p)
        """
        if risk_reward_ratio <= 0:
            return 0.0
            
        b = risk_reward_ratio
        p = win_probability
        q = 1 - p
        
        kelly_percentage = (b * p - q) / b
        
        # Apply fractional Kelly
        position_size_pct = max(0.0, kelly_percentage * self.kelly_fraction)
        
        # Cap position size (e.g., max 5% of account per trade)
        MAX_POS_SIZE = 0.05
        position_size_pct = min(position_size_pct, MAX_POS_SIZE)
        
        return account_balance * position_size_pct

    def check_stop_loss(self, entry_price: float, current_price: float, position_type: str, sl_pct: float = 0.02) -> bool:
        """
        Returns True if stop loss hit.
        """
        if position_type == "LONG":
            if current_price <= entry_price * (1 - sl_pct):
                return True
        elif position_type == "SHORT":
            if current_price >= entry_price * (1 + sl_pct):
                return True
        return False
