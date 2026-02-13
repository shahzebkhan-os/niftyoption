import pytest
import numpy as np
from src.features.volatility import black_scholes_greeks, find_iv

def test_bs_reference_values():
    """Verify BS price and delta against known reference values."""
    S, K, T, r, sigma = 100, 100, 1, 0.05, 0.2
    price, delta, gamma, theta, vega = black_scholes_greeks(S, K, T, r, sigma, 'CE')
    
    # Expected values for ATM Call: Price ~10.45, Delta ~0.63
    assert price == pytest.approx(10.4505, abs=1e-4)
    assert delta == pytest.approx(0.6368, abs=1e-4)
    assert gamma > 0
    assert vega > 0

def test_near_zero_expiry():
    """Ensure stability when T is very small."""
    S, K, T, r, sigma = 100, 100, 1e-9, 0.05, 0.2
    price, delta, _, _, _ = black_scholes_greeks(S, K, T, r, sigma, 'CE')
    
    # At T=0, ATM Call should be 0, Delta should be 0.5 (limit) or 1/0 depending on implementation
    # Current implementation uses max(T, eps) where eps = 1e-7
    assert price >= 0
    assert not np.isnan(price)

def test_deep_itm_otm():
    """Verify Delta limits for deep ITM and OTM options."""
    # Deep OTM Call
    p_otm, d_otm, _, _, _ = black_scholes_greeks(100, 200, 1, 0.05, 0.2, 'CE')
    assert d_otm == pytest.approx(0, abs=1e-2)
    
    # Deep ITM Call
    p_itm, d_itm, _, _, _ = black_scholes_greeks(100, 50, 1, 0.05, 0.2, 'CE')
    assert d_itm == pytest.approx(1, abs=1e-2)

def test_iv_solver_convergence():
    """Verify that find_iv can recover the original sigma."""
    S, K, T, r, sigma_true = 100, 100, 0.5, 0.05, 0.25
    market_price, _, _, _, _ = black_scholes_greeks(S, K, T, r, sigma_true, 'CE')
    
    sigma_recovered = find_iv(market_price, S, K, T, r, 'CE')
    assert sigma_recovered == pytest.approx(sigma_true, abs=1e-3)

def test_zero_volatility_edge_case():
    """Handle sigma=0 gracefully."""
    S, K, T, r, sigma = 100, 100, 1, 0.05, 0
    price, delta, _, _, _ = black_scholes_greeks(S, K, T, r, sigma, 'CE')
    assert price >= 0
    assert not np.isnan(price)
