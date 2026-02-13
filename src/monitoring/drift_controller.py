import logging

logger = logging.getLogger(__name__)

class DriftController:
    def __init__(self):
        self.state = "STABLE"

    def evaluate(self, performance_metrics, calibration_score, feature_drift):
        """
        Evaluates the system state based on multiple monitoring inputs.
        
        Args:
            performance_metrics: dict from PerformanceDriftMonitor.compare()
            calibration_score: float (Brier score from CalibrationMonitor)
            feature_drift: dict of p-values from FeatureDriftMonitor.detect_drift()
        """
        # 1. Check for Severe Performance Degradation
        # If Sharpe falls below 60% of backtest, we trigger severe warning
        if performance_metrics.get("sharpe_ratio_vs_bt", 1.0) < 0.6:
            self.state = "SEVERE_DRIFT"
            logger.critical("SYSTEM ALERT: Severe Performance Drift Detected. Throttling Capital.")

        # 2. Check for Structural Feature Shift
        elif any(p < 0.01 for p in feature_drift.values()):
            self.state = "FEATURE_DRIFT"
            logger.warning("SYSTEM ALERT: Structural Feature Shift Detected. Reducing Exposure.")

        # 3. Check for Model Calibration Issues
        # Brier score > 0.25 typically means the model is no better than guessing
        elif calibration_score > 0.25:
            self.state = "CALIBRATION_DRIFT"
            logger.warning("SYSTEM ALERT: Model Calibration Drift Detected.")

        else:
            self.state = "STABLE"
            logger.info("System health: STABLE")

        return self.state

    def capital_multiplier(self):
        """
        Returns the risk-adjusted capital scale based on the system state.
        """
        mapping = {
            "STABLE": 1.0,
            "CALIBRATION_DRIFT": 0.75,
            "FEATURE_DRIFT": 0.5,
            "SEVERE_DRIFT": 0.25 # Or 0.0 to pause entirely
        }
        multiplier = mapping.get(self.state, 1.0)
        
        if multiplier < 1.0:
            logger.info(f"Risk Throttle Active: Capital Scale = {multiplier:.2f} (State: {self.state})")
            
        return multiplier
