import pandas as pd
import numpy as np
import logging
import os
from datetime import datetime, timedelta
from sqlalchemy import text
from src.data.database import get_engine
from src.features.feature_pipeline import build_features
from src.models.regime_trainer import RegimeTrainer
from src.models.target import TargetGenerator
from src.config.settings import settings

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TrainingPipeline:
    def __init__(self, symbol="NIFTY", horizon_mins=1440, threshold=100):
        self.symbol = symbol
        self.horizon_mins = horizon_mins
        self.threshold = threshold
        self.engine = get_engine()
        self.target_gen = TargetGenerator(horizon_minutes=horizon_mins, threshold_points=threshold)
        
        # We now use specialized trainers per regime
        self.regimes = ["TRENDING_UP", "TRENDING_DOWN", "VOLATILITY_EXPANSION", "RANGE_BOUND"]

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
        Uses the STEP 2 & 3 Institutional Engine to build a robust model input stream.
        """
        if df.empty:
            return pd.DataFrame()

        logger.info(f"Processing {len(df)} records with Institutional Feature Engine...")
        
        # 1. Standardize and Build Features (includes regime classification & lag-safety)
        df_features = build_features(df, lag_safety=True)
        
        if df_features.empty:
            return pd.DataFrame()

        # 2. Generate Targets
        logger.info("Generating horizon-based targets...")
        df_features['target'] = self.target_gen.generate_target(
            df_features, 
            price_col='underlying_price', 
            time_col='timestamp'
        )
        
        return df_features.dropna(subset=['target'])

    def run_training(self, start_date, end_date):
        """Orchestrates the training of regime-specific models."""
        raw_df = self.load_raw_data(start_date, end_date)
        if len(raw_df) < 500:
            logger.error(f"Insufficient data ({len(raw_df)}). Need more history.")
            return

        processed_df = self.preprocess_and_feature_engineer(raw_df)
        if processed_df.empty:
            logger.error("No features generated. Check data density.")
            return

        # Feature columns defined in Step 2 & 3
        # Note: We exclude 'regime_encoded' from specialized models because they are already 
        # trained on data subsetted by regime. We also remove confidence/stability from X 
        # to focus on core predictive signal.
        feature_cols = [
            'iv_percentile', 'iv_zscore', 'oi_velocity', 'oi_acceleration', 
            'divergence', 'trap_score', 'net_gex',
            'delta', 'gamma', 'theta', 'vega', 'iv'
        ]
        
        # Train specialized model for each regime
        trained_models = []
        for regime in self.regimes:
            logger.info(f"--- Training Orchestration: {regime} ---")
            trainer = RegimeTrainer(regime_name=regime)
            model = trainer.train(processed_df, feature_cols, 'target')
            if model:
                trained_models.append(regime)

        logger.info(f"Multi-regime ensemble training cycle complete. Models trained: {trained_models}")
        return trained_models

if __name__ == "__main__":
    # Example usage: Train on last 30 days
    end = datetime.now()
    start = end - timedelta(days=180) # 6 months
    
    pipeline = TrainingPipeline(symbol=settings.SYMBOLS[0])
    pipeline.run_training(start, end)
