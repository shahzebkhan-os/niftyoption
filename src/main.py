import asyncio
import logging
import os
import numpy as np
import pandas as pd
from datetime import datetime
from src.config.settings import settings
from src.utils.logger import setup_logging
from src.data.market_data import MarketDataFetcher
from src.data.database import get_session, OptionChainSnapshot, init_db
from src.features.volatility import VolatilityFeatures
from src.features.greeks import DealerPositioningFeatures
from src.features.trend import TrendFeatures
from src.features.order_flow import OrderFlowFeatures
from src.strategy.regime_classifier import RegimeClassifier
from src.strategy.risk_manager import RiskManager
from src.strategy.telegram_bot import TelegramManager
from src.models.prediction_model import ModelTrainer

# Setup Logging
logger = setup_logging()

class OptionsEngine:
    def __init__(self):
        self.fetcher = MarketDataFetcher(symbol=settings.SYMBOL, interval=settings.INTERVAL_SECONDS)
        self.telegram = TelegramManager(token=settings.TELEGRAM_BOT_TOKEN, chat_id=settings.TELEGRAM_CHAT_ID)
        self.risk_manager = RiskManager()
        self.regime_classifier = RegimeClassifier()
        self.dealer_features = DealerPositioningFeatures()
        self.vol_features = VolatilityFeatures()
        self.trend_features = TrendFeatures()
        self.order_flow = OrderFlowFeatures()
        
        # Load Model
        self.model = ModelTrainer()
        self.load_latest_model()

    def load_latest_model(self):
        """Loads the most recent model for the symbol."""
        try:
            model_dir = "models"
            if not os.path.exists(model_dir):
                logger.warning("Models directory not found. Using mock predictions.")
                return
                
            models = [f for f in os.listdir(model_dir) if f.startswith(f"calibrated_lgb_{settings.SYMBOL}") and f.endswith(".pkl")]
            if not models:
                logger.warning(f"No models found for {settings.SYMBOL}. Using mock.")
                return
                
            latest_model = sorted(models)[-1]
            path = os.path.join(model_dir, latest_model)
            self.model.load_model(path)
            logger.info(f"Loaded model: {path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")

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
        """Calculates features for the latest snapshot."""
        # 1. Underlying time-series for technicals
        underlying_ts = df.groupby('timestamp')['underlying_price'].first().reset_index()
        underlying_ts.columns = ['timestamp', 'close']
        underlying_ts = underlying_ts.sort_values('timestamp')
        
        # EMA distances
        for p in [9, 21, 50]:
            underlying_ts[f'ema_{p}'] = underlying_ts['close'].ewm(span=p, adjust=False).mean()
        
        latest = underlying_ts.iloc[-1].copy()
        ema_dist_9_21 = (latest['ema_9'] - latest['ema_21']) / latest['ema_21']
        ema_dist_21_50 = (latest['ema_21'] - latest['ema_50']) / latest['ema_50']
        
        # Volatility
        returns = underlying_ts['close'].pct_change()
        realized_vol = returns.rolling(window=30).std().iloc[-1] * np.sqrt(375)
        
        # 2. Snapshot-based features (GEX, OI)
        latest_ts = df['timestamp'].max()
        current_snap = df[df['timestamp'] == latest_ts]
        gex = self.dealer_features.calculate_net_gex(current_snap, latest['close'])
        hhi = self.order_flow.calculate_strike_concentration(current_snap)
        
        return pd.DataFrame([{
            'ema_dist_9_21': ema_dist_9_21,
            'ema_dist_21_50': ema_dist_21_50,
            'realized_vol': realized_vol,
            'net_gex': gex.get('net_gex', 0),
            'oi_hhi': hhi
        }])

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
            
            # Regime Detection (Legacy logic preserved)
            underlying_ts = df.groupby('timestamp')['underlying_price'].first().reset_index()
            underlying_ts.columns = ['timestamp', 'close']
            regime = self.regime_classifier.detect_regime(underlying_ts.sort_values('timestamp'), 
                                                          volatility_features={'iv_percentile': 50})
            
            # Prediction
            try:
                if self.model.calibrated_model:
                    prob = self.model.predict_proba(features_df)[0]
                    logger.info(f"Model Prediction: {prob:.4f}")
                else:
                    prob = 0.65 # Mock fallback
            except Exception as e:
                logger.error(f"Prediction failed: {e}")
                prob = 0.0
            
            # Signal Generation
            if prob > settings.SIGNAL_THRESHOLD: # Threshold in settings
                signal_data = {
                    'regime': regime,
                    'symbol': settings.SYMBOL,
                    'action': 'BUY CALL' if regime == 'TRENDING_UP' else 'STRATEGY_SIGNAL',
                    'strike': int(df['underlying_price'].iloc[0] // 50 * 50), # ATM Approx
                    'expiry': df['expiry'].iloc[0],
                    'confidence': prob * 100,
                    'ev': 10.5, # Placeholder for EV model
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
