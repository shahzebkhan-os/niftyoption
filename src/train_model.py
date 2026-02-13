import pandas as pd
import numpy as np
import logging
import asyncio
import os
from datetime import datetime, timedelta
from sqlalchemy import text
from src.data.database import get_engine
from src.features.trend import TrendFeatures
from src.features.volatility import VolatilityFeatures
from src.features.greeks import DealerPositioningFeatures
from src.features.order_flow import OrderFlowFeatures
from src.models.prediction_model import ModelTrainer
from src.models.target import TargetGenerator
from src.config.settings import settings

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TrainingPipeline:
    def __init__(self, symbol="NIFTY", horizon_mins=30, threshold=20):
        self.symbol = symbol
        self.horizon_mins = horizon_mins
        self.threshold = threshold
        self.engine = get_engine()
        self.target_gen = TargetGenerator(horizon_minutes=horizon_mins, threshold_points=threshold)
        self.trainer = ModelTrainer()
        
        # Feature generators
        self.trend = TrendFeatures()
        self.vol = VolatilityFeatures()
        self.dealer = DealerPositioningFeatures()
        self.order_flow = OrderFlowFeatures()

    def load_raw_data(self, start_date, end_date):
        """Loads historical snapshots from DB."""
        logger.info(f"Loading raw data for {self.symbol} from {start_date} to {end_date}")
        query = f"""
        SELECT * FROM option_chain_snapshots 
        WHERE symbol = '{self.symbol}' 
        AND timestamp >= '{start_date}' 
        AND timestamp <= '{end_date}'
        AND underlying_price IS NOT NULL
        ORDER BY timestamp ASC
        """
        with self.engine.connect() as conn:
            df = pd.read_sql(text(query), conn)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df

    def preprocess_and_feature_engineer(self, df):
        """Standardizes time-series and calculates features."""
        if df.empty:
            return pd.DataFrame()

        logger.info(f"Preprocessing {len(df)} records...")
        
        # 1. Aggregate to unique timestamps for underlying price
        underlying_df = df.groupby('timestamp').agg({
            'underlying_price': 'first',
            'volume': 'sum'
        }).reset_index()
        underlying_df.columns = ['timestamp', 'close', 'volume']
        
        # 2. Resample to 1-minute intervals to ensure consistency
        underlying_df.set_index('timestamp', inplace=True)
        # Note: If data is sparse, 'ffill' is used but might introduce bias for very many missing bars.
        # For training, we prefer high density.
        underlying_df = underlying_df.resample('1min').ffill()
        
        # 3. Individual Snapshot processing (for Greeks/OI)
        # We'll calculate aggregate GEX/Concentration per timestamp
        logger.info("Calculating snapshot-based features (GEX, OI concentration)...")
        snap_features = []
        for ts, group in df.groupby('timestamp'):
            gex = self.dealer.calculate_net_gex(group, group['underlying_price'].iloc[0])
            hhi = self.order_flow.calculate_strike_concentration(group)
            snap_features.append({
                'timestamp': ts,
                'net_gex': gex.get('net_gex', 0),
                'oi_hhi': hhi
            })
        
        snap_df = pd.DataFrame(snap_features)
        snap_df['timestamp'] = pd.to_datetime(snap_df['timestamp'])
        snap_df.set_index('timestamp', inplace=True)
        snap_df = snap_df.resample('1min').ffill()
        
        # 4. Merge
        final_df = underlying_df.join(snap_df, how='left').ffill()
        
        # 5. Calculate derived features
        logger.info("Computing technical features...")
        # EMAs
        for p in [9, 21, 50]:
            final_df[f'ema_{p}'] = final_df['close'].ewm(span=p, adjust=False).mean()
        
        # Trend Score
        # (Need to implement rolling trend score if possible, or just use EMA distances)
        final_df['ema_dist_9_21'] = (final_df['ema_9'] - final_df['ema_21']) / final_df['ema_21']
        final_df['ema_dist_21_50'] = (final_df['ema_21'] - final_df['ema_50']) / final_df['ema_50']
        
        # Volatility
        final_df['returns'] = final_df['close'].pct_change()
        final_df['realized_vol'] = final_df['returns'].rolling(window=30).std() * np.sqrt(375) # Daily scale approx
        
        # Target
        logger.info("Generating targets...")
        final_df['target'] = self.target_gen.generate_target(final_df.reset_index(), price_col='close', time_col='timestamp').values
        
        return final_df.dropna()

    def run_training(self, start_date, end_date):
        """Runs the full pipeline."""
        raw_df = self.load_raw_data(start_date, end_date)
        if len(raw_df) < 1000:
            logger.error("Insufficient data for training.")
            return

        processed_df = self.preprocess_and_feature_engineer(raw_df)
        if processed_df.empty:
            logger.error("Failed to generate features.")
            return

        # Prepare X, y
        feature_cols = ['ema_dist_9_21', 'ema_dist_21_50', 'realized_vol', 'net_gex', 'oi_hhi']
        X = processed_df[feature_cols]
        y = processed_df['target']
        
        logger.info(f"Training on {len(X)} samples with {len(feature_cols)} features.")
        
        # Walk forward validation
        scores = self.trainer.walk_forward_validation(X, y)
        avg_auc = np.mean([s['auc'] for s in scores])
        logger.info(f"Walk-Forward Average AUC: {avg_auc:.4f}")
        
        # Final training on all data
        model = self.trainer.train_calibrated(X, y)
        
        # Save
        if not os.path.exists('models'):
            os.makedirs('models')
            
        model_path = f"models/calibrated_lgb_{self.symbol}_{datetime.now().strftime('%Y%m%d')}.pkl"
        self.trainer.save_model(model_path)
        logger.info(f"Model saved to {model_path}")
        
        # Update latest simlink or settings? 
        return model_path

if __name__ == "__main__":
    # Example usage: Train on last 30 days
    end = datetime.now().strftime('%Y-%m-%d')
    start = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    pipeline = TrainingPipeline(symbol=settings.SYMBOL)
    pipeline.run_training(start, end)
