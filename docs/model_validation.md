# Model Validation & Stability

The Options Intelligence Engine employs a rigorous validation hierarchy to ensure model reliability in non-stationary markets.

## 1. Stability Audit (Cross-Validation)
- **Temporal Split**: `TimeSeriesSplit` with 4 folds is used for all training.
- **Overfitting Guard**: Models are rejected if:
    - Mean AUC < 0.51 (Poor performance).
    - AUC Std > 0.15 (High variance/overfitting).
    - Foundational drift detected in feature distributions.

## 2. Calibration
- **Isotonic Regression**: Used via `CalibratedClassifierCV` to ensure predicted probabilities map to actual frequencies.
- **Reliability Checks**: Periodic calibration audits to ensure `prob=0.8` actually means 80% outcome success.

## 3. Drift Governance (DriftController)
- **Feature Drift**: Monitoring JS Divergence and PSI for core predictive features.
- **Performance Drift**: Tracking realized vs. expected PnL and Max Drawdown.
- **Governance States**:
    - `GOVERNANCE_NOMINAL`: Full capital deployment.
    - `MILD_DRIFT`: Capital scaled down by 25%.
    - `SEVERE_DRIFT`: System-wide trading pause.

## 4. Reproducibility
- **Fixed Seeds**: `random_state=42` enforced across all sklearn/LightGBM components.
- **Metadata Registry**: Every model version is stored with its training context, metrics, and feature set.
