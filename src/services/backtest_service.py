import asyncio
import concurrent.futures
import pandas as pd
import numpy as np
import logging
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta
from src.strategy.regime_classifier import RegimeClassifier
from src.strategy.risk_manager import RiskManager
from src.data.database import get_engine, get_session, OptionChainSnapshot
from sqlalchemy import text

logger = logging.getLogger(__name__)

class BacktestService:
    def __init__(self):
        self.regime_classifier = RegimeClassifier()
        self.risk_manager = RiskManager()
        
    async def run_backtest(self, config: dict):
        """
        Main entry point for running a backtest session.
        config keys: start_date, end_date, interval, indicators, risk, execution, model_id
        """
        logger.info(f"Initializing backtest for {config.get('symbol', 'NIFTY')}...")
        
        # 1. Load Data
        df = await self._load_data(config)
        if df.empty:
            return {"error": "Insufficient data for selected range."}
            
        # 2. Resample & Prep Features
        df = self._prepare_data(df, config)
        
        # 3. Execution (Event-Driven Loop)
        # Shift simulation to ProcessPool for heavy lifting if range is large
        results = self._simulate(df, config)
        
        # 4. Save to DB for comparison
        self._save_results(results, config)
        
        return results

    async def _load_data(self, config):
        """Loads historical option chain and underlying data from DB."""
        start = config.get('start_date')
        end = config.get('end_date')
        symbol = config.get('symbol', 'NIFTY')
        
        engine = get_engine()
        # Query for underlying spot prices primarily (or most liquid ATM)
        # Assuming we need a 'clean' time-series of the underlying
        # Make end date inclusive by adding 1 day or using < next day
        try:
            end_dt = datetime.strptime(end, '%Y-%m-%d') + timedelta(days=1)
            end_str = end_dt.strftime('%Y-%m-%d')
        except:
            end_str = end

        query = f"""
        SELECT timestamp, MAX(underlying_price) as close, SUM(volume) as volume, 
               AVG(underlying_price) as open, MAX(underlying_price) as high, MIN(underlying_price) as low
        FROM option_chain_snapshots
        WHERE symbol = '{symbol}' 
          AND timestamp >= '{start}' 
          AND timestamp < '{end_str}'
          AND underlying_price IS NOT NULL
        GROUP BY timestamp
        ORDER BY timestamp ASC
        """
        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            return df

    def _prepare_data(self, df, config):
        """Advanced feature engineering with custom parameterization."""
        interval = config.get('interval', '1m')
        ind = config.get('indicators', {})
        
        # 1. Resample
        df = df.resample(interval).agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).ffill()
        
        # 2. Institutional Indicators
        # EMA Stack
        for period in ind.get('ema_periods', [9, 21, 50]):
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
            
        # ATR window
        df['atr'] = self._calculate_atr(df, ind.get('atr_window', 14))
        
        # OI Velocity Proxy (if data available)
        # Note: In real setup, we'd query OI separately from the snapshots
        df['oi_velocity'] = df['volume'].rolling(window=ind.get('oi_smoothing', 5)).mean().pct_change()
        
        # GEX / Trap Detection Proxy Logic
        df['trap_score'] = (df['close'] - df['ema_21']).abs() * ind.get('trap_sensitivity', 1.0)
        
        return df

    def _calculate_atr(self, df, window):
        high_low = df['high'] - df['low']
        high_pc = (df['high'] - df['close'].shift()).abs()
        low_pc = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_pc, low_pc], axis=1).max(axis=1)
        return tr.rolling(window=window).mean()

    async def run_parameter_sweep(self, base_config: dict, sweep_options: dict):
        """Runs multiple backtests across a grid of parameters using multi-processing."""
        # 1. Generate combinations
        ema_stack = sweep_options.get('ema_periods', [[9, 21], [21, 50], [9, 50]])
        risk_stack = sweep_options.get('risk_multipliers', [1.0, 1.5, 2.0])
        
        combinations = []
        for ema in ema_stack:
            for risk in risk_stack:
                cfg = base_config.copy()
                cfg['indicators'] = cfg['indicators'].copy()
                cfg['indicators']['ema_periods'] = ema
                cfg['risk'] = cfg['risk'].copy()
                cfg['risk']['risk_multiplier'] = risk
                combinations.append(cfg)
        
        logger.info(f"Starting Parameter Sweep: {len(combinations)} combinations.")
        
        # 2. Load Data Once
        df = await self._load_data(base_config)
        if df.empty: return {"error": "No data for sweep."}
        
        # 3. Parallel Execution
        # ProcessPoolExecutor requires picklable objects. 
        # We'll map a sync wrapper of the simulation.
        import concurrent.futures
        loop = asyncio.get_running_loop()
        
        results = []
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # Note: _simulate is already synchronous, which is good for ProcessPool
            futures = [executor.submit(self._run_single_sweep, df, comb) for comb in combinations]
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
                
        # 4. Rank by Sharpe
        results = sorted(results, key=lambda x: x['metrics']['sharpe'], reverse=True)
        return results

    def _run_single_sweep(self, df, config):
        """Helper to run a single backtest in a process worker."""
        # Pre-prep features for this config
        df_prep = self._prepare_data(df, config)
        res = self._simulate(df_prep, config)
        res['config'] = config
        return res

    def _simulate(self, df, config):
        """Internal event-driven simulation engine."""
        exec_config = config.get('execution', {})
        risk_config = config.get('risk', {})
        
        # Enforce determinism
        seed = config.get('random_seed', 42)
        np.random.seed(seed)
        
        capital = risk_config.get('initial_capital', 100000)
        initial_capital = capital
        slippage = exec_config.get('slippage', 0.0005)
        fee = exec_config.get('fee', 0.0003)
        
        # Risk Configuration
        max_positions = risk_config.get('max_open_positions', 1)
        kelly_fraction = risk_config.get('kelly_fraction', 0.2) # Default to 0.2 for conservative growth
        
        # Simulation State
        equity_curve = []
        trades = []
        position = None
        cooldown = 0
        pending_signal = None
        open_positions = [] # Track concurrent positions
        
        # Risk State
        daily_pnl = 0
        last_date = None
        max_daily_loss = risk_config.get('max_daily_loss_pct', 0.02) * initial_capital

        # Simulation Loop
        for i in range(len(df)):
            row = df.iloc[i]
            timestamp = df.index[i]
            
            # Reset daily pnl on new day
            cur_date = timestamp.date()
            if cur_date != last_date:
                daily_pnl = 0
                last_date = cur_date

            # Update Equity & Handle Exits
            cur_price = float(row['close'])
            ec_val = capital
            
            exited_positions = []
            for pos in open_positions:
                # 1. Handle Exits
                exit_price, reason = self._check_exit(row, pos, slippage)
                if exit_price:
                    pnl_pct = (exit_price - pos['entry']) / pos['entry'] if pos['direction'] == 'LONG' else \
                              (pos['entry'] - exit_price) / pos['entry']
                    pnl_raw = pnl_pct * pos['size']
                    costs = (pos['size'] * fee) + (pos['size'] * (1 + pnl_pct) * fee)
                    net_pnl = pnl_raw - costs
                    
                    capital += net_pnl
                    daily_pnl += net_pnl
                    
                    trades.append({
                        'entry_ts': pos['entry_ts'],
                        'exit_ts': timestamp,
                        'entry': pos['entry'],
                        'exit': exit_price,
                        'net_pnl': net_pnl,
                        'costs': costs,
                        'reason': reason,
                        'pnl': net_pnl # Added for win_rate metric
                    })
                    exited_positions.append(pos)
                else:
                    # Unrealized PnL for equity curve
                    pnl = (cur_price - pos['entry']) / pos['entry'] if pos['direction'] == 'LONG' else \
                          (pos['entry'] - cur_price) / pos['entry']
                    ec_val += pnl * pos['size']
            
            for pos in exited_positions:
                open_positions.remove(pos)
                
            equity_curve.append({'ts': timestamp, 'equity': ec_val})
            
            # 2. Daily Drawdown Check
            if daily_pnl < -max_daily_loss:
                continue

            # 3. Handle Pending Entries (1-bar delay)
            if pending_signal and len(open_positions) < max_positions:
                entry_price = float(row['open']) * (1 + slippage) if pending_signal['direction'] == 'LONG' else \
                              float(row['open']) * (1 - slippage)
                
                direction = pending_signal['direction']
                
                # --- Fractional Kelly Sizing ---
                p = float(pending_signal['prob'])
                win_loss_ratio = 2.0
                kelly_f = (p * win_loss_ratio - (1 - p)) / win_loss_ratio
                kelly_f = max(0, min(1, kelly_f))
                
                pos_size = capital * kelly_f * kelly_fraction
                
                sl_dist = float(row['atr']) * 2 if 'atr' in row and not pd.isna(row['atr']) else cur_price * 0.01
                
                new_pos = {
                    'entry_ts': timestamp,
                    'entry': entry_price,
                    'size': pos_size,
                    'stop': entry_price - sl_dist if direction == 'LONG' else entry_price + sl_dist,
                    'target': entry_price + (sl_dist * 2) if direction == 'LONG' else entry_price - (sl_dist * 2),
                    'direction': direction,
                    'prob': p
                }
                open_positions.append(new_pos)
                pending_signal = None
                continue

            # 4. Signal Detection (T -> T+1 entry)
            if cooldown > 0:
                cooldown -= 1
                continue
                
            # Signal Probe (Integration Placeholder)
            prob = np.random.uniform(0.4, 0.8) 
            if prob > 0.65:
                pending_signal = {
                    'direction': 'LONG',
                    'prob': prob
                }

        metrics = self._calculate_metrics(initial_capital, capital, equity_curve, trades)
        
        # 5. Export Results (Phase 5: Backtest Reporting)
        self._export_results(metrics, equity_curve, trades)
        
        return {
            "metrics": metrics,
            "equity_curve": equity_curve,
            "trades": trades
        }

    def _check_exit(self, row, pos, slippage):
        """Handles gap realism and stop/target checks."""
        if pos['direction'] == 'LONG':
            if row['open'] < pos['stop']: # Gap down
                return row['open'] * (1 - slippage), "STOP (GAP)"
            if row['low'] <= pos['stop']:
                return pos['stop'] * (1 - slippage), "STOP"
            if row['high'] >= pos['target']:
                return pos['target'] * (1 - slippage), "TARGET"
        else:
            if row['open'] > pos['stop']: # Gap up
                return row['open'] * (1 + slippage), "STOP (GAP)"
            if row['high'] >= pos['stop']:
                return pos['stop'] * (1 + slippage), "STOP"
            if row['low'] <= pos['target']:
                return pos['target'] * (1 + slippage), "TARGET"
        return None, None

    def _calculate_metrics(self, initial, final, ec, trades):
        """
        Calculates production-grade risk-adjusted performance metrics.
        """
        if not ec or not trades:
            return {"total_return_pct": 0, "sharpe": 0, "total_trades": 0}

        df_ec = pd.DataFrame(ec).set_index('ts')
        returns = df_ec['equity'].pct_change().dropna()
        
        total_return = (final - initial) / initial
        
        # Sharpe (Annualized)
        # Assuming 1min data, 375 mins per day
        risk_free_rate = 0.05
        excess_returns = returns - (risk_free_rate / (252 * 375))
        sharpe = (excess_returns.mean() / returns.std() * np.sqrt(252 * 375)) if len(returns) > 0 and returns.std() != 0 else 0
        
        # Sortino
        downside_returns = returns[returns < 0]
        sortino = (excess_returns.mean() / downside_returns.std() * np.sqrt(252 * 375)) if len(downside_returns) > 0 and downside_returns.std() != 0 else 0
        
        # Drawdown
        mdd = (df_ec['equity'] / df_ec['equity'].cummax() - 1).min()
        
        # CAGR
        days = (df_ec.index[-1] - df_ec.index[0]).days
        if days > 0:
            cagr = ((final / initial) ** (365.0 / days)) - 1
        else:
            cagr = 0
            
        # Win Rate & Profit Factor
        winning_trades = [t for t in trades if t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] <= 0]
        
        win_rate = len(winning_trades) / len(trades)
        
        gross_profit = sum(t['pnl'] for t in winning_trades)
        gross_loss = abs(sum(t['pnl'] for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else gross_profit
        
        expectancy = (win_rate * (gross_profit/len(winning_trades) if winning_trades else 0)) + \
                     ((1 - win_rate) * (sum(t['pnl'] for t in losing_trades)/len(losing_trades) if losing_trades else 0))

        return {
            "total_return_pct": float(total_return * 100),
            "cagr_pct": float(cagr * 100),
            "sharpe": float(sharpe),
            "sortino": float(sortino),
            "max_drawdown_pct": float(mdd * 100),
            "win_rate_pct": float(win_rate * 100),
            "profit_factor": float(profit_factor),
            "expectancy": float(expectancy),
            "total_trades": len(trades)
        }

    def _export_results(self, metrics, ec, trades):
        """Export artifacts for external analysis."""
        import os
        import json
        
        output_dir = "backtest_results"
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(output_dir, f"run_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)
        
        # Save metrics
        with open(os.path.join(run_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=4)
            
        # Save trades
        pd.DataFrame(trades).to_csv(os.path.join(run_dir, "trades.csv"), index=False)
        
        # Save equity curve
        pd.DataFrame(ec).to_csv(os.path.join(run_dir, "equity_curve.csv"), index=False)
        
        logger.info(f"Backtest artifacts exported to {run_dir}")

    def _save_results(self, results, config):
        """Stub for DB persistence."""
        pass
