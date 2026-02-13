import pytest
import pandas as pd
import numpy as np
from src.services.backtest_service import BacktestService

def test_backtest_determinism():
    """Verify that identical seeds produce identical results."""
    bs = BacktestService()
    
    # Mock data
    dates = pd.date_range(start="2026-01-01", periods=100, freq="1min")
    df = pd.DataFrame({
        'open': np.linspace(100, 110, 100),
        'high': np.linspace(101, 111, 100),
        'low': np.linspace(99, 109, 100),
        'close': np.linspace(100.5, 110.5, 100),
        'volume': [100] * 100,
        'ema_21': [100] * 100, # Needed for trap_score
        'atr': [1] * 100
    }, index=dates)
    
    config = {
        'random_seed': 42,
        'risk': {'initial_capital': 100000},
        'execution': {'slippage': 0.001, 'fee': 0.0005}
    }
    
    # Run twice
    res1 = bs._simulate(df, config)
    res2 = bs._simulate(df, config)
    
    # Assert metrics are identical
    assert res1['metrics']['total_return_pct'] == res2['metrics']['total_return_pct']
    assert len(res1['trades']) == len(res2['trades'])
    
    # Assert specific trade details are identical
    if res1['trades']:
        assert res1['trades'][0]['net_pnl'] == res2['trades'][0]['net_pnl']

def test_backtest_different_seeds():
    """Verify that different seeds produce different results."""
    bs = BacktestService()
    
    dates = pd.date_range(start="2026-01-01", periods=100, freq="1min")
    df = pd.DataFrame({
        'open': np.linspace(100, 110, 100),
        'high': np.linspace(101, 111, 100),
        'low': np.linspace(99, 109, 100),
        'close': np.linspace(100.5, 110.5, 100),
        'volume': [100] * 100,
        'ema_21': [100] * 100,
        'atr': [1] * 100
    }, index=dates)
    
    res1 = bs._simulate(df, {'random_seed': 42, 'risk': {'initial_capital': 100000}})
    res2 = bs._simulate(df, {'random_seed': 99, 'risk': {'initial_capital': 100000}})
    
    # With different seeds, it's highly likely (though not guaranteed for tiny datasets) 
    # that results differ due to the random signal probe
    # If they are still the same, we might need a more volatile mock signal
    assert res1['metrics']['total_return_pct'] != res2['metrics']['total_return_pct'] or \
           len(res1['trades']) != len(res2['trades'])
