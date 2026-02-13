from src.services.backtest_service import BacktestService
import logging

logger = logging.getLogger(__name__)

class BacktestRouter:
    """
    Service Router for Backtest operations. 
    This can be easily wrapped by a FastAPI router if needed.
    """
    def __init__(self):
        self.service = BacktestService()

    async def run_standard_backtest(self, config: dict):
        """Triggers a high-fidelity single backtest run."""
        logger.info("Routing standard backtest request...")
        return await self.service.run_backtest(config)

    async def run_optimized_sweep(self, base_config: dict, sweep_options: dict):
        """Triggers a multi-core parameter optimization sweep."""
        logger.info("Routing parameter sweep request...")
        return await self.service.run_parameter_sweep(base_config, sweep_options)

    def export_results(self, results: dict, format: str = "csv"):
        """Handles data formatting for exports (CSV, JSON)."""
        if format == "csv":
            import pandas as pd
            df = pd.DataFrame(results.get("trades", []))
            return df.to_csv(index=False)
        return results
