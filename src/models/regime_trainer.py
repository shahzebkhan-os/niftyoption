import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
import joblib
import os
import pandas as pd
import logging
import numpy as np

logger = logging.getLogger(__name__)

class RegimeTrainer:
    def __init__(self, regime_name, model_dir="models"):
        self.regime_name = regime_name
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

    def train(self, df, feature_columns, target_column):
        """
        Trains and calibrates a model specifically for the given regime.
        Includes a stability audit across folds to detect overfitting.
        """
        # Filter data for regime only
        df_regime = df[df["regime"] == self.regime_name]
        
        if len(df_regime) < 100: 
            logger.warning(f"Insufficient data for regime {self.regime_name}.")
            return None

        X = df_regime[feature_columns]
        y = df_regime[target_column]

        # 1. Stability Audit (Manual Cross-Validation)
        # Reducing to 4 splits given the limited number of unique trading days
        tscv = TimeSeriesSplit(n_splits=4)
        fold_auc = []
        
        logger.info(f"Auditing stability for {self.regime_name} across 4 folds...")
        for train_idx, val_idx in tscv.split(X):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Use smaller estimators and shallow trees to detect foundational instability
            fold_model = lgb.LGBMClassifier(
                n_estimators=50, 
                learning_rate=0.1, 
                max_depth=3, 
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42, # Enforce reproducibility
                verbose=-1
            )
            fold_model.fit(X_tr, y_tr)
            
            from sklearn.metrics import roc_auc_score
            preds = fold_model.predict_proba(X_val)[:, 1]
            try:
                fold_auc.append(roc_auc_score(y_val, preds))
            except:
                continue # Skip folds with only one class
        
        if not fold_auc:
            logger.warning(f"REJECTED: No valid AUC folds for {self.regime_name}.")
            return None

        auc_std = np.std(fold_auc)
        mean_auc = np.mean(fold_auc)
        logger.info(f"Stability Audit: Mean AUC={mean_auc:.4f}, Std={auc_std:.4f}")

        # Final stability check
        if np.isnan(mean_auc) or (len(fold_auc) > 1 and auc_std > 0.15) or mean_auc < 0.51:
            logger.warning(f"REJECTED: Model stability/performance too low (Std={auc_std:.4f}, Mean={mean_auc:.4f}).")
            return None
        
        # 2. Final Calibrated Training
        base_model = lgb.LGBMClassifier(
            n_estimators=200,
            learning_rate=0.03,
            max_depth=3,
            reg_alpha=1.0,
            reg_lambda=1.0,
            random_state=42, # Enforce reproducibility
            importance_type='gain',
            verbose=-1
        )

        calibrated_model = CalibratedClassifierCV(
            base_model,
            method="isotonic",
            cv=tscv
        )

        calibrated_model.fit(X, y)

        # 3. Structured Saving (Phase 3: Reproducibility)
        import json
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_dir = os.path.join(self.model_dir, f"{self.regime_name}_{timestamp}")
        os.makedirs(version_dir, exist_ok=True)
        
        model_path = os.path.join(version_dir, "model.pkl")
        joblib.dump(calibrated_model, model_path)
        
        metadata = {
            "regime": self.regime_name,
            "version": timestamp,
            "timestamp": datetime.now().isoformat(),
            "features": feature_columns,
            "target": target_column,
            "metrics": {
                "mean_auc": float(mean_auc),
                "auc_std": float(auc_std),
                "num_samples": len(df_regime)
            },
            "random_state": 42
        }
        
        with open(os.path.join(version_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=4)
            
        logger.info(f"Model saved to {version_dir} with metadata.")
        
        # Legacy support: also save to root model_dir
        legacy_path = os.path.join(self.model_dir, f"{self.regime_name}_model.pkl")
        joblib.dump(calibrated_model, legacy_path)
        
        return calibrated_model
