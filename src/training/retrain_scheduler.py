import schedule
import time
import logging
import asyncio
import pandas as pd
from datetime import datetime, timedelta
from src.training.walk_forward_trainer import WalkForwardTrainer
from src.train_model import TrainingPipeline
from src.config.settings import settings

logger = logging.getLogger(__name__)

class RetrainOrchestrator:
    def __init__(self, symbol=None):
        self.symbol = symbol or settings.SYMBOLS[0]
        self.trainer = WalkForwardTrainer()
        self.pipeline = TrainingPipeline(symbol=self.symbol)

    def run_retraining_cycle(self):
        """
        Executes the full automated retraining cycle.
        """
        logger.info(f"--- Starting Scheduled Retraining Cycle for {self.symbol} ---")
        try:
            # 1. Load Data (Last 12 months for a rich walk-forward window)
            end = datetime.now()
            start = end - timedelta(days=365)
            
            logger.info("Loading historical data for retraining...")
            raw_df = self.pipeline.load_raw_data(start, end)
            
            if len(raw_df) < 1000:
                logger.warning("Insufficient data for a meaningful retraining cycle.")
                return

            # 2. Build Features
            logger.info("Building features and regimes...")
            processed_df = self.pipeline.preprocess_and_feature_engineer(raw_df)
            
            if processed_df.empty:
                logger.error("Feature engineering failed during retraining.")
                return

            # 3. Run Walk-Forward Orchestrator
            feature_cols = [
                'iv_percentile', 'iv_zscore', 'oi_velocity', 'oi_acceleration', 
                'divergence', 'trap_score', 'net_gex'
            ]
            
            # This triggers training, evaluation, gating, and promotion
            results = self.trainer.run_all_regimes(processed_df, feature_cols, 'target')
            
            promoted = [r for r, p in results.items() if p is not None]
            logger.info(f"Cycle Complete. Promoted regimes: {promoted}")
            
        except Exception as e:
            logger.error(f"Retraining cycle failed: {e}", exc_info=True)

def start_scheduler():
    orchestrator = RetrainOrchestrator()
    
    # Schedule: Every Sunday at 02:00 AM
    schedule.every().sunday.at("02:00").do(orchestrator.run_retraining_cycle)
    
    logger.info("Retraining Scheduler ONLINE (Sunday 02:00 AM)")
    
    # Also run once on startup for verification if needed (optional)
    # orchestrator.run_retraining_cycle()

    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    start_scheduler()
