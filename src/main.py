import asyncio
import logging
import os
import numpy as np
import pandas as pd
from datetime import datetime
from src.config.settings import settings
from src.utils.logger import setup_logging
from src.data.market_fetcher import MarketDataFetcher
from src.data.database import get_session, OptionChainSnapshot, init_db
from src.features.feature_pipeline import build_features
from src.strategy.regime_classifier import RegimeClassifier
from src.strategy.risk_manager import RiskManager
from src.strategy.telegram_bot import TelegramManager
from src.models.model_registry import ModelRegistry
from src.models.ensemble import RegimeEnsemble

# Setup Logging
logger = setup_logging()

class OptionsEngine:
    def __init__(self):
        self.fetcher = MarketDataFetcher(symbol=settings.SYMBOL, interval=settings.INTERVAL_SECONDS)
        self.telegram = TelegramManager(token=settings.TELEGRAM_BOT_TOKEN, chat_id=settings.TELEGRAM_CHAT_ID)
        self.risk_manager = RiskManager()
        self.regime_classifier = RegimeClassifier()
        
        # Load Model Registry and Ensemble
        self.registry = ModelRegistry()
        self.ensemble = RegimeEnsemble(self.registry)

    def load_latest_model(self):
        """No longer used. Models are managed by registry."""
        pass

    async def run(self):
        logger.info("Starting Options Intelligence Engine...")
        
        # Ensure DB is ready
        try:
            init_db()
        except Exception as e:
            logger.critical(f"FATAL: Database initialization failed: {e}")
            return
        
        # Start Data Fetcher in background
        fetcher_task = asyncio.create_task(self.fetcher.run_loop())
        
        # Main Strategy Loop
        logger.info("Engine initialized. Entering main loop.")
        while True:
            try:
                await self.process_cycle()
                await asyncio.sleep(settings.INTERVAL_SECONDS) # Run strategy every minute
            except asyncio.CancelledError:
                logger.info("Engine stopping...")
                fetcher_task.cancel()
                break
            except Exception as e:
                logger.error(f"Error in strategy loop: {e}", exc_info=True)
                await asyncio.sleep(settings.INTERVAL_SECONDS)

    def prepare_inference_features(self, df):
        """
        Uses the STEP 2 Institutional Feature Engine to build inference inputs.
        Ensures that live predictions use the exact same lag-safe logic as training.
        """
        if df.empty:
            return pd.DataFrame()

        # Build features (includes internal lag-safety/shifting)
        # Note: For live inference, we need the LATEST row that has features.
        # build_features(df, lag_safety=True) shifts features forward.
        # So row T will have features from T-1.
        df_features = build_features(df, lag_safety=True)
        
        if df_features.empty:
            return pd.DataFrame()

        # Extract the latest observation for inference
        feature_cols = [
            'iv_percentile', 'iv_zscore', 'oi_velocity', 'oi_acceleration', 
            'divergence', 'trap_score', 'net_gex',
            'regime_encoded', 'regime_confidence', 'regime_stability'
        ]
        
        return df_features.tail(1)[feature_cols]

    async def process_cycle(self):
        logger.info("Processing cycle...")
        session = get_session()
        try:
            # Fetch last 1000 snapshots for context
            query = session.query(OptionChainSnapshot).order_by(OptionChainSnapshot.timestamp.desc()).limit(1000)
            df = pd.read_sql(query.statement, session.bind)
            
            if df.empty:
                logger.warning("No data to process.")
                return

            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.dropna(subset=['underlying_price'])

            if len(df.groupby('timestamp')) < 50:
                logger.warning("Insufficient history for indicators.")
                return

            # Feature Calculation
            features_df = self.prepare_inference_features(df)
            
            # Prediction via Regime Ensemble
            try:
                # Extract components for ensemble
                latest_features = features_df.iloc[0]
                
                # Core features for the specialized models
                core_feature_cols = [
                    'iv_percentile', 'iv_zscore', 'oi_velocity', 'oi_acceleration', 
                    'divergence', 'trap_score', 'net_gex'
                ]
                core_features = latest_features[core_feature_cols].values
                
                # Regime info (from the processed feature stream)
                # We need the original regime string, but build_features added 'regime' column
                # Let's ensure 'regime' is available or use encoded mapping
                # Actually, our regime_df merge in build_features added 'regime' (string)
                regime = latest_features.get('regime', 'RANGE_BOUND')
                regime_conf = latest_features.get('regime_confidence', 1.0)
                
                prob = self.ensemble.predict(core_features, regime, regime_conf)
                logger.info(f"Regime: {regime} | Conf: {regime_conf:.2f} | prob: {prob:.4f}")

            except Exception as e:
                logger.error(f"Ensemble prediction failed: {e}")
                prob = 0.5
            
            # Signal Generation
            if prob > settings.SIGNAL_THRESHOLD: 
                signal_data = {
                    'regime': latest_features.get('regime', 'N/A'),
                    'symbol': settings.SYMBOL,
                    'action': 'IDENTIFIED_MOVE',
                    'strike': int(df['underlying_price'].iloc[0] // 50 * 50),
                    'expiry': df['expiry'].iloc[0],
                    'confidence': prob * 100,
                    'regime_stability': latest_features.get('regime_stability', 0.0),
                    'risk_level': 'MEDIUM'
                }
                await self.telegram.send_alert(signal_data)
                
        except Exception as e:
            logger.error(f"Cycle failed: {e}", exc_info=True)
        finally:
            session.close()

if __name__ == "__main__":
    engine = OptionsEngine()
    try:
        asyncio.run(engine.run())
    except KeyboardInterrupt:
        logger.info("Engine stopped by user (SIGINT).")
