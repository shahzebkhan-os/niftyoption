import pandas as pd
import numpy as np
import logging
from src.strategy.regime_classifier import RegimeClassifier
from src.strategy.risk_manager import RiskManager
from src.models.prediction_model import ModelTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Backtester:
    def __init__(self, start_date, end_date, initial_capital=100000, slippage=0.0005, fee=0.0003):
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.equity_curve = []
        self.trades = []
        self.position = None  # {entry, size, stop, target, direction}
        self.slippage = slippage # 0.05%
        self.fee = fee # 0.03%
        
        self.regime_classifier = RegimeClassifier()
        self.risk_manager = RiskManager()
        self.model = ModelTrainer() # Load trained model
        
        # Cooldown state
        self.cooldown_counter = 0
        self.consecutive_losses = 0

    def run(self, data: pd.DataFrame):
        """
        Replays the strategy over historical data row-by-row.
        data: DataFrame with features, prices (open, high, low, close).
        """
        logger.info("Starting Institutional Backtest (Event-Driven)...")
        
        # Ensure we have OHLC for realistic stop checks
        if 'high' not in data.columns: data['high'] = data['close']
        if 'low' not in data.columns: data['low'] = data['close']
        if 'open' not in data.columns: data['open'] = data['close']
        
        for i in range(len(data)):
            row = data.iloc[i]
            timestamp = data.index[i]
            
            # 1. Update Equity (Mark to Market)
            current_equity = self.capital
            if self.position:
                # Unrealized PnL
                price = row['close']
                if self.position['direction'] == 'LONG':
                    unrealized = (price - self.position['entry']) * (self.position['size'] / self.position['entry'])
                else: 
                    unrealized = (self.position['entry'] - price) * (self.position['size'] / self.position['entry'])
                current_equity += unrealized
            
            self.equity_curve.append({'timestamp': timestamp, 'equity': current_equity})

            # 2. Manage Existing Position
            if self.position:
                self._manage_position(row)
                continue # If in position, skip entry logic (assuming 1 position max)

            # 3. Cooldown Check
            if self.cooldown_counter > 0:
                self.cooldown_counter -= 1
                continue

            # 4. Detect Regime & Signal
            # Use data up to i (simulate real-time)
            # Efficiently: pass current row features or small window
            # For backtest speed, we assume features are pre-calculated in 'data'
            
            regime = self.regime_classifier.detect_regime(data.iloc[max(0, i-50):i+1])
            
            # Filter: Only trade in favorable regimes
            if regime not in ['TRENDING_UP', 'TRENDING_DOWN', 'high_iv']: 
                # Example filter
                pass

            # 5. Prediction
            prob = self.get_prediction(row)

            # 6. Entry Logic
            if prob > 0.6:
                direction = 'LONG' # Logic based on prob or other signal
                self._open_position(row, prob, direction, regime)

        self.calculate_metrics()

    def get_prediction(self, row):
        # Placeholder for model call. 
        # In production: return self.model.predict(row)
        # Here: Mock random probability for simulation
        return np.random.uniform(0.4, 0.9)

    def _open_position(self, row, prob, direction, regime):
        # Entry Price with Slippage
        # Long: Buy at Ask (Close + slip)
        price = row['close']
        entry_price = price * (1 + self.slippage)
        
        # Kelly Sizing
        # Kelly % = (b*p - q) / b
        # Risk:Reward = 2.0 (Target 2%, Stop 1%)
        position_size = self.risk_manager.calculate_position_size(self.capital, prob, 2.0)
        
        # Stop & Target
        # ATR based or fixed %
        # Example: 1% Stop, 2% Target
        stop_input = self.risk_manager.check_stop_loss(entry_price, 0, direction, 0.01) # Check logic inside
        # Actually risk manager returns bool, we need price.
        # Let's calculate inline for now or improve RiskManager
        
        sl_pct = 0.01
        tp_pct = 0.02
        
        if direction == 'LONG':
            stop_loss = entry_price * (1 - sl_pct)
            target = entry_price * (1 + tp_pct)
        else:
            stop_loss = entry_price * (1 + sl_pct)
            target = entry_price * (1 - tp_pct)

        self.position = {
            'entry': entry_price,
            'size': position_size,
            'stop': stop_loss,
            'target': target,
            'direction': direction,
            'regime': regime
        }

    def _manage_position(self, row):
        # Check High/Low for hits first
        # Conservative: Check Stop on Low, Target on High (for Long)
        
        # Assuming intra-minute price action:
        # If Low <= Stop -> Stopped Out
        # If High >= Target -> Take Profit
        # Conflict? (Both hit)? Assume Stop hit first if Bearish candle? 
        
        pos = self.position
        exit_price = None
        exit_reason = None
        
        if pos['direction'] == 'LONG':
            if row['open'] < pos['stop']: # Gap Down
                exit_price = row['open'] * (1 - self.slippage)
                exit_reason = 'STOP_LOSS (GAP)'
            elif row['low'] <= pos['stop']:
                exit_price = pos['stop'] * (1 - self.slippage)
                exit_reason = 'STOP_LOSS'
            elif row['high'] >= pos['target']:
                # Basic check: did we hit target? 
                # Realism: if Open > Target (Gap Up), exit at Open
                if row['open'] > pos['target']:
                    exit_price = row['open'] * (1 - self.slippage)
                    exit_reason = 'TAKE_PROFIT (GAP)'
                else:
                    exit_price = pos['target'] * (1 - self.slippage)
                    exit_reason = 'TAKE_PROFIT'
            elif row['close'] <= pos['stop']: # Close check fallback
                exit_price = row['close'] * (1 - self.slippage)
                exit_reason = 'STOP_LOSS'
        
        if exit_price:
            # Calculate PnL
            # Size is Amount Invested
            # Return = (Exit - Entry) / Entry
            pnl_pct = (exit_price - pos['entry']) / pos['entry']
            pnl_gross = pnl_pct * pos['size']
            
            # Costs
            commission = pos['size'] * self.fee + (pos['size'] * (1+pnl_pct)) * self.fee
            pnl_net = pnl_gross - commission
            
            self.capital += pnl_net
            self.trades.append({
                'entry': pos['entry'],
                'exit': exit_price,
                'pnl': pnl_net,
                'reason': exit_reason,
                'regime': pos['regime']
            })
            
            # Update Cooldown
            if pnl_net < 0:
                self.consecutive_losses += 1
                if self.consecutive_losses >= 2:
                    self.cooldown_counter = 5 # Pause for 5 bars
            else:
                self.consecutive_losses = 0
                
            self.position = None

    def calculate_metrics(self):
        equity_df = pd.DataFrame(self.equity_curve)
        if equity_df.empty: 
            return

        # Resample to Daily for Standard Sharpe (Assuming 'timestamp' is datetime)
        equity_df.set_index('timestamp', inplace=True)
        # Handle duplicate indices or intraday
        daily_equity = equity_df['equity'].resample('D').last().dropna()
        
        if len(daily_equity) < 2:
            daily_returns = equity_df['equity'].pct_change().dropna() # Fallback to minute
            period_factor = 252 * 375 # Minute annualized
            logger.warning("Not enough daily data. Using minute-based Sharpe.")
        else:
            daily_returns = daily_equity.pct_change().dropna()
            period_factor = 252 # Daily annualized

        total_return = (self.capital - self.initial_capital) / self.initial_capital
        
        sharpe = 0
        sortino = 0
        if daily_returns.std() != 0:
            sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(period_factor)
            
            # Sortino (Downside Deviation)
            downside = daily_returns[daily_returns < 0]
            if len(downside) > 0:
                sortino = (daily_returns.mean() / downside.std()) * np.sqrt(period_factor)
        
        max_drawdown = (equity_df['equity'] / equity_df['equity'].cummax() - 1).min()
        
        logger.info(f"--- Backtest Results ---")
        logger.info(f"Final Capital: {self.capital:.2f}")
        logger.info(f"Total Return: {total_return*100:.2f}%")
        logger.info(f"Sharpe Ratio: {sharpe:.2f}")
        logger.info(f"Sortino Ratio: {sortino:.2f}")
        logger.info(f"Max Drawdown: {max_drawdown*100:.2f}%")
        logger.info(f"Total Trades: {len(self.trades)}")
        
        # Win Rate
        wins = [t for t in self.trades if t['pnl'] > 0]
        win_rate = len(wins) / len(self.trades) if self.trades else 0
        logger.info(f"Win Rate: {win_rate*100:.1f}%")

if __name__ == "__main__":
    # Mock Data with OHLC
    np.random.seed(42)
    dates = pd.date_range(start="2023-01-01", periods=2000, freq="1min")
    vals = np.random.randn(2000)
    # create a trend
    vals[500:1000] += 0.5 # Uptrend
    vals[1500:1800] -= 0.5 # Downtrend
    
    close = vals.cumsum() + 10000
    high = close + np.abs(np.random.randn(2000)) * 2
    low = close - np.abs(np.random.randn(2000)) * 2
    open_p = close + np.random.randn(2000) * 1 
    
    df = pd.DataFrame({
        'open': open_p,
        'high': high,
        'low': low,
        'close': close,
        'volume': np.random.randint(1000, 100000, 2000)
    }, index=dates)
    
    # Calculate EMAs for Regime Detection
    df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    
    backtester = Backtester("2023-01-01", "2023-01-02")
    backtester.run(df)
