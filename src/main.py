import asyncio
import logging
import os
import numpy as np
import pandas as pd
from datetime import datetime
from sqlalchemy import text
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
from src.monitoring.performance_drift import PerformanceDriftMonitor
from src.monitoring.calibration_monitor import CalibrationMonitor
from src.monitoring.feature_drift import FeatureDriftMonitor
from src.monitoring.drift_controller import DriftController

# Setup Logging
logger = setup_logging()

class OptionsEngine:
    def __init__(self):
        self.fetcher = MarketDataFetcher(symbol=settings.SYMBOLS[0], interval=settings.INTERVAL_SECONDS)
        self.telegram = TelegramManager(token=settings.TELEGRAM_BOT_TOKEN, chat_id=settings.TELEGRAM_CHAT_ID)
        self.risk_manager = RiskManager()
        self.regime_classifier = RegimeClassifier()
        
        # Load Model Registry and Ensemble
        self.registry = ModelRegistry()
        self.ensemble = RegimeEnsemble(self.registry)

        # Drift Governance Suite
        self.drift_controller = DriftController()
        self.perf_monitor = PerformanceDriftMonitor()
        self.calib_monitor = CalibrationMonitor()
        self.feature_monitor = FeatureDriftMonitor()
        
        # Performance Tracking State
        self.live_trades = []
        self.live_equity = [100000.0] # Starting capital
        self.historical_data_sample = pd.DataFrame() 
        
        # Production Safety State (Phase 6)
        self.daily_pnl = 0.0
        self.last_pnl_reset = datetime.now().date()
        self.last_alert_time = {} # For throttling
        self.max_daily_loss = 0.02 * self.live_equity[0] # 2% hard stop
        self.max_positions = 3

    async def initialize_monitoring_data(self):
        """Seeds the monitors with historical benchmarks from the DB."""
        try:
            logger.info("Initializing drift monitoring benchmarks...")
            # Load last 1000 snapshots as a historical baseline for feature drift
            with get_session() as session:
                query = text("SELECT * FROM option_chain_snapshots ORDER BY timestamp DESC LIMIT 1000")
                self.historical_data_sample = pd.read_sql(query, session.bind)
                self.historical_data_sample['timestamp'] = pd.to_datetime(self.historical_data_sample['timestamp'])
            logger.info(f"Loaded {len(self.historical_data_sample)} baseline records.")
        except Exception as e:
            logger.error(f"Failed to initialize monitoring data: {e}")

    def prepare_inference_features(self, df: pd.DataFrame):
        """Standardized feature preparation for live inference."""
        if df.empty:
            return pd.DataFrame()

        df_features = build_features(df, lag_safety=True)
        
        if df_features.empty:
            return pd.DataFrame()

        feature_cols = [
            'iv_percentile', 'iv_zscore', 'oi_velocity', 'oi_acceleration', 
            'divergence', 'trap_score', 'net_gex',
            'regime_encoded', 'regime_confidence', 'regime_stability', 'regime'
        ]
        
        # Ensure we only return columns that exist
        available_cols = [c for c in feature_cols if c in df_features.columns]
        return df_features.tail(1)[available_cols]

    async def process_cycle(self):
        logger.info("Processing cycle...")
        if self.historical_data_sample.empty:
            await self.initialize_monitoring_data()

        for symbol in settings.SYMBOLS:
            logger.info(f"--- Processing {symbol} ---")
            # 1. Fetch Latest Data for specific symbol
            self.fetcher.symbol = symbol
            df = await self.fetcher.get_latest_data()
            
            # Fallback to DB if fetcher is empty
            if df.empty:
                with get_session() as session:
                    query = session.query(OptionChainSnapshot).filter(OptionChainSnapshot.symbol == symbol).order_by(OptionChainSnapshot.timestamp.desc()).limit(200)
                    df = pd.read_sql(query.statement, session.bind)
            
            if df.empty:
                logger.warning(f"No data snapshots collected for {symbol}.")
                continue

            # 2. Prep Features
            features_df = self.prepare_inference_features(df)
            if features_df.empty:
                continue
                
            # 3. Prediction via Regime Ensemble
            try:
                latest_features = features_df.iloc[0]
                core_feature_cols = ['iv_percentile', 'iv_zscore', 'oi_velocity', 'oi_acceleration', 'divergence', 'trap_score', 'net_gex']
                
                # Check if all core features are present
                if not all(col in latest_features for col in core_feature_cols):
                    logger.warning(f"Missing core features for prediction in {symbol}.")
                    continue

                core_features = latest_features[core_feature_cols].values
                regime = latest_features.get('regime', 'RANGE_BOUND')
                regime_conf = latest_features.get('regime_confidence', 1.0)
                
                prob = self.ensemble.predict(core_features, regime, regime_conf)
                logger.info(f"{symbol} | Regime: {regime} | Conf: {regime_conf:.2f} | prob: {prob:.4f}")

                # 4. 🧠 DRIFT GOVERNANCE LAYER
                feature_report = self.feature_monitor.detect_drift(self.historical_data_sample, features_df, core_feature_cols)
                perf_report = self.perf_monitor.compare(self.live_trades, self.live_equity)
                calib_score = 0.0 # Placeholder
                
                state = self.drift_controller.evaluate(perf_report, calib_score, feature_report)
                capital_scale = self.drift_controller.capital_multiplier()

                if state == "SEVERE_DRIFT":
                    logger.critical(f"SYSTEM PAUSED for {symbol}: Severe drift detected.")
                    continue

                # Signal Generation
                if prob > settings.SIGNAL_THRESHOLD: 
                    # Circuit Breaker: Model Confidence Stability (Phase 6)
                    if regime_conf < 0.4:
                        logger.warning(f"CIRCUIT BREAKER: Low regime confidence ({regime_conf:.2f}) for {symbol}. Signal suppressed.")
                        continue

                    final_prob = prob * capital_scale
                    
                    if final_prob > settings.SIGNAL_THRESHOLD:
                        # Daily Loss Guard (Phase 6)
                        today = datetime.now().date()
                        if today != self.last_pnl_reset:
                            self.daily_pnl = 0.0
                            self.last_pnl_reset = today
                            
                        if self.daily_pnl <= -self.max_daily_loss:
                            logger.warning(f"SAFETY GUARD: Daily loss limit hit ({self.daily_pnl:.2f}). No new signals.")
                            return

                        signal_data = {
                            'regime': regime,
                            'symbol': symbol,
                            'action': 'IDENTIFIED_MOVE',
                            'strike': int(df['underlying_price'].iloc[0] // 50 * 50),
                            'expiry': df['expiry'].iloc[0],
                            'confidence': final_prob * 100,
                            'governance_state': state,
                            'risk_level': 'MEDIUM',
                            'is_dry_run': settings.DRY_RUN
                        }
                        
                        # Alert Throttling (Phase 6: Max 1 alert per 15 mins per regime)
                        throttle_key = f"{regime}_{symbol}_{signal_data['strike']}"
                        now = datetime.now()
                        if throttle_key in self.last_alert_time:
                            if (now - self.last_alert_time[throttle_key]).total_seconds() < 900:
                                logger.info(f"THROTTLED: Signal for {throttle_key} suppressed.")
                                continue
                        
                        self.last_alert_time[throttle_key] = now
                        
                        if settings.DRY_RUN:
                            logger.info(f"[DRY_RUN] Signal Generated for {symbol}: {signal_data}")
                        else:
                            await self.telegram.send_alert(signal_data)

            except Exception as e:
                logger.error(f"Cycle execution failed for {symbol}: {e}", exc_info=True)

    async def run(self):
        logger.info("Starting Options Intelligence Engine...")
        init_db()
        await self.initialize_monitoring_data()
        
        await self.telegram.send_message(f"🚀 Options Intelligence Engine ONLINE (Symbols: {', '.join(settings.SYMBOLS)})")
        
        while True:
            try:
                await self.process_cycle()
                await asyncio.sleep(settings.INTERVAL_SECONDS)
            except Exception as e:
                logger.error(f"Main loop error: {e}")
                await asyncio.sleep(5)

if __name__ == "__main__":
    engine = OptionsEngine()
    try:
        asyncio.run(engine.run())
    except KeyboardInterrupt:
        logger.info("Engine stopped by user (SIGINT).")
