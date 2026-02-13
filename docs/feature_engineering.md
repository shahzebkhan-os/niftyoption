# Feature Engineering Guide

The engine employs a sophisticated feature pipeline to extract institutional Alpha from raw option chain data.

## 1. Volatility Dynamics
- **Implied Volatility (IV)**: Derived using a Newton-Raphson solver for the Black-Scholes model.
- **IV Percentile**: 60-day rolling percentile of IV, used to identify volatility extremes.
- **IV Z-Score**: Standardized IV relative to its 60-day mean, highlighting statistical anomalies.

## 2. Institutional Greek Exposure
- **Delta/Gamma/Theta/Vega**: Standard Black-Scholes Greeks.
- **Net GEX (Gamma Exposure)**: Aggregated gamma across the option chain, used to identify market "gravity" or potential breakout zones.

## 3. Orderflow & Positioning
- **OI Velocity**: Rate of change of Open Interest, proxy for institutional position accumulation.
- **OI Acceleration**: Second-order change of OI, used to detect pre-breakout surges.
- **Divergence**: Price vs. OI divergence, identifying weakening trends or potential reversals.

## 4. Proprietary Signals
- **Trap Score**: Measures price deviation from the 21-bar EMA weighted by volatility, used to identify over-extended "traps."
- **Net GEX Divergence**: Identifying when institutional positioning doesn't match the price action.

## Lag Safety Protocol
To prevent look-ahead bias, all features are strictly lagged:
- `Feature[T]` is computed using data available at `T`.
- `Prediction[T]` uses `Feature[T-1]`.
- This ensures 100% reproducibility and realism in backtests.
