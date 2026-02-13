# Options Intelligence Engine 🚀

A production-grade quantitative trading infrastructure for institutional-grade options analysis and automated strategy execution.

## 🏗 Key Components
- ** Institutional Feature Engine**: IV Greeks, GEX, and Trap scores with verified lag-safety.
- ** Regime-Switching Ensemble**: Specialized LightGBM models for Trending Up, Trending Down, Volatility Expansion, and Range-Bound environments.
- ** Production Hardened**: Integrated risk manager, daily loss guards, and circuit breakers.
- ** Drift Governance**: Automated monitoring for feature and performance drift.

## 🛠 Quickstart

### 1. Environment Setup
```bash
make install
cp .env.example .env
# Update .env with your credentials (Groww JWT, Telegram Bot, Postgres)
```

### 2. Development Workflow
Standardized tasks via `Makefile`:
- `make lint`: Run ruff/black checks.
- `make test`: Execute the full test suite (75%+ coverage).
- `make run`: Launch the live engine (defaults to DRY_RUN).
- `make backtest`: Run simulation over historical data.
- `make dashboard`: Launch the analysis UI.

## 🛡 Production Safety Philosophy
- **Dry Run First**: All systems default to `DRY_RUN=True` to prevent accidental execution.
- **Daily Loss Guards**: Automatic trading halt if PnL drops below 2% of equity.
- **Model Stability Audit**: Models must pass stability cross-validation (AUC Std < 0.15) before being registered.
- **Throttled Alerts**: Prevent signal spamming with 15-minute cool-down periods.

## 📚 Documentation
- [Architecture Overview](docs/architecture.md)
- [Feature Engineering Detail](docs/feature_engineering.md)
- [Model Validation Process](docs/model_validation.md)

## ⚖ License
MIT License.
