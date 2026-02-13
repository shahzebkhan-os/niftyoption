import pandas as pd
import numpy as np
import logging
import asyncio
import os
from datetime import datetime, timedelta
from sqlalchemy import text
from src.data.database import get_engine
from src.features.feature_pipeline import build_features
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
        """
        Uses the STEP 2 Institutional Feature Engine to build a robust model input stream.
        """
        if df.empty:
            return pd.DataFrame()

        logger.info(f"Processing {len(df)} records with Institutional Feature Engine...")
        
        # 1. Standardize and Build Features (includes lag-safety)
        # Note: build_features handles resampling/interpolation via groupby(timestamp)
        # but here we pass the raw snapshots directly as it handles contract-level logic first.
        df_features = build_features(df, lag_safety=True)
        
        if df_features.empty:
            return pd.DataFrame()

        # 2. Generate Targets
        # Targets are generated at the contract/timestamp level
        # IMPORTANT: Target generation uses FUTURE info (look-ahead), 
        # but the MODEL input (features) is already shifted (lag-safe).
        logger.info("Generating horizon-based targets...")
        df_features['target'] = self.target_gen.generate_target(
            df_features.reset_index(), 
            price_col='underlying_price', 
            time_col='timestamp'
        ).values
        
        return df_features.dropna()

    def run_training(self, start_date, end_date):
        """Runs the full pipeline."""
        raw_df = self.load_raw_data(start_date, end_date)
        if len(raw_df) < 500: # Threshold for training
            logger.error(f"Insufficient data ({len(raw_df)}). Need more history.")
            return

        processed_df = self.preprocess_and_feature_engineer(raw_df)
        if processed_df.empty:
            logger.error("No features generated. Check data density.")
            return

        # Prepare X, y
        # Using the feature columns defined in the STEP 2 pipeline
        feature_cols = [
            'iv_percentile', 'iv_zscore', 'oi_velocity', 'oi_acceleration', 
            'divergence', 'trap_score', 'net_gex'
        ]
        
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
