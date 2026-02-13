import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class PerformanceDriftMonitor:
    def __init__(self, backtest_metrics=None):
        """
        backtest_metrics: dict containing 'sharpe' and 'winrate' from historical simulations.
        """
        self.backtest_metrics = backtest_metrics or {"sharpe": 2.0, "winrate": 0.55}

    def rolling_sharpe(self, equity_curve, window=50):
        """
        Calculates the annualized rolling Sharpe Ratio.
        """
        if len(equity_curve) < window:
            return pd.Series([0.0])
            
        returns = pd.Series(equity_curve).pct_change().dropna()
        rolling = returns.rolling(window)
        # Annualization factor: sqrt(252 * trading_periods_per_day)
        # Assuming intraday, using a simplified 252 for daily-equivalent.
        sharpe = (rolling.mean() / rolling.std()) * np.sqrt(252)
        return sharpe

    def winrate(self, trades):
        """
        Calculates winrate from a list of trade objects with 'pnl' key.
        """
        if not trades:
            return 0.0
        wins = [t for t in trades if t.get("pnl", 0) > 0]
        return len(wins) / len(trades)

    def compare(self, live_trades, live_equity):
        """
        Compares live metrics against backtest benchmarks.
        """
        if not live_equity or len(live_equity) < 2:
            return {
                "live_sharpe": 0.0,
                "live_winrate": 0.0,
                "sharpe_ratio_vs_bt": 1.0,
                "winrate_diff": 0.0
            }

        live_sharpe_series = self.rolling_sharpe(live_equity)
        live_sharpe = live_sharpe_series.iloc[-1] if not live_sharpe_series.empty else 0.0
        live_winrate = self.winrate(live_trades)

        sharpe_drift = live_sharpe / self.backtest_metrics["sharpe"] if self.backtest_metrics["sharpe"] != 0 else 1.0
        winrate_drift = live_winrate - self.backtest_metrics["winrate"]

        return {
            "live_sharpe": live_sharpe,
            "live_winrate": live_winrate,
            "sharpe_ratio_vs_bt": sharpe_drift,
            "winrate_diff": winrate_drift
        }
