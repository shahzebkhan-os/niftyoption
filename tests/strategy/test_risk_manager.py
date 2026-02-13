import pytest
from src.strategy.risk_manager import RiskManager

def test_kelly_calculation():
    """Verify Kelly calculation for known win/loss probabilities."""
    rm = RiskManager(kelly_fraction=1.0)
    
    # 60% win prob, 1:1 risk/reward
    # Kelly % = (1 * 0.6 - 0.4) / 1 = 0.2
    size = rm.calculate_position_size(100000, 0.6, 1.0)
    assert size == 100000 * 0.05 # Capped at 5% (0.05) by MAX_POS_SIZE

def test_max_risk_cap():
    """Ensure position size never exceeds the hard cap."""
    rm = RiskManager(kelly_fraction=1.0)
    
    # Very high win prob should still be capped
    size = rm.calculate_position_size(100000, 0.9, 2.0)
    # Kelly % = (2*0.9 - 0.1)/2 = 0.85 -> Capped to 0.05
    assert size == 100000 * 0.05

def test_negative_expectancy():
    """Ensure size is 0 for negative expectancy trades."""
    rm = RiskManager()
    
    # 40% win prob, 1:1 risk/reward
    # Kelly % = (1*0.4 - 0.6)/1 = -0.2 -> 0.0
    size = rm.calculate_position_size(100000, 0.4, 1.0)
    assert size == 0.0

def test_stop_loss_hit():
    """Verify stop loss detection."""
    rm = RiskManager()
    
    # Long position, 2% stop loss
    assert rm.check_stop_loss(100, 97, "LONG", sl_pct=0.02) is True
    assert rm.check_stop_loss(100, 99, "LONG", sl_pct=0.02) is False
    
    # Short position, 2% stop loss
    assert rm.check_stop_loss(100, 103, "SHORT", sl_pct=0.02) is True
    assert rm.check_stop_loss(100, 101, "SHORT", sl_pct=0.02) is False
