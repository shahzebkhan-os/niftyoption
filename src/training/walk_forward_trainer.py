import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime, timedelta
from src.models.regime_trainer import RegimeTrainer
from src.training.model_evaluator import ModelEvaluator
from src.training.model_versioning import ModelVersionManager
from src.models.model_registry import ModelRegistry

logger = logging.getLogger(__name__)

class WalkForwardTrainer:
    def __init__(self, window_months=6, min_auc_gain=0.01):
        """
        window_months: How much historical data to use for the walk-forward window.
        min_auc_gain: Minimum AUC improvement required to promote a new model.
        """
        self.window_months = window_months
        self.min_auc_gain = min_auc_gain
        self.evaluator = ModelEvaluator()
        self.version_manager = ModelVersionManager()
        self.registry = ModelRegistry() # Used to get the current benchmark models

    def run_regime_cycle(self, df, regime, feature_columns, target_column):
        """
        Orchestrates retraining for a specific regime.
        """
        logger.info(f"--- Walk-Forward Cycle: {regime} ---")
        
        # 1. Prepare Window Data
        latest_date = df['timestamp'].max()
        start_date = latest_date - pd.DateOffset(months=self.window_months)
        df_window = df[df['timestamp'] >= start_date]
        
        # 2. Train New Model Candidate
        trainer = RegimeTrainer(regime_name=regime, model_dir="models/candidates")
        new_model = trainer.train(df_window, feature_columns, target_column)
        
        if new_model is None:
            return None

        # 3. Evaluate New Candidate
        # Note: We evaluate on the SAME window for comparison
        new_metrics = self.evaluator.evaluate(new_model, df_window[df_window['regime'] == regime], 
                                               feature_columns, target_column)
        logger.info(f"New Model metrics: AUC={new_metrics['auc']:.4f}, Brier={new_metrics['brier']:.4f}")

        # 4. Benchmarking & Gating
        current_model = self.registry.get_model(regime)
        if current_model:
            current_metrics = self.evaluator.evaluate(current_model, df_window[df_window['regime'] == regime], 
                                                     feature_columns, target_column)
            logger.info(f"Current Model metrics: AUC={current_metrics['auc']:.4f}, Brier={current_metrics['brier']:.4f}")
            
            # GATING LOGIC: Deploy only if improved
            if new_metrics['auc'] > (current_metrics['auc'] + self.min_auc_gain):
                logger.info(f"PROMOTION GRANTED: Candidate improves on benchmark ({new_metrics['auc']:.4f} > {current_metrics['auc']:.4f})")
                
                # Promote via Version Manager
                temp_path = os.path.join("models/candidates", f"{regime}_model.pkl")
                final_path = self.version_manager.promote_model(regime, temp_path)
                return final_path
            else:
                logger.warning(f"PROMOTION REJECTED: Candidate failed to beat benchmark significantly.")
                return None
        else:
            # First time training this regime, promote immediately
            logger.info("PROMOTION GRANTED: No existing model found. First deployment.")
            temp_path = os.path.join("models/candidates", f"{regime}_model.pkl")
            return self.version_manager.promote_model(regime, temp_path)

    def run_all_regimes(self, df, feature_columns, target_column):
        """Runs the walk-forward retraining for all regimes detected in data."""
        regimes = df['regime'].unique()
        results = {}
        for regime in regimes:
            path = self.run_regime_cycle(df, regime, feature_columns, target_column)
            results[regime] = path
            
        return results
