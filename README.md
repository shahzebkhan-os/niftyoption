# Intraday Options Intelligence Engine

## 🧠 System Thesis
This project is a production-grade, capital-preserving intelligence engine designed to predict significant intraday moves in equity indices (NIFTY/BANKNIFTY) using high-frequency options chain data. It prioritizes **probabilistic edge** over high-frequency turnover, utilizing a regime-adaptive ensemble approach.

---

## 🏗 Architectural Blueprint (AI-Context)
The system is built as a modular, asynchronous pipeline divided into five distinct layers.

### 1. Data Ingestion Layer (`src/data/`)
-   **GrowwClient**: Asynchronous `aiohttp` client. Handles fragmented JSON responses, session persistence, and rate-limiting backoff.
-   **MarketDataFetcher**: Continuous polling loop that serializes full options chain snapshots.
-   **NSE FO Bhavcopy Loader**: Robust ingestion pipeline for historical data. Handles ZIP extraction, holiday parsing, and institutional-grade bulk upserts with `ON CONFLICT` deduplication.
-   **Database**: PostgreSQL storage utilizing `SQLAlchemy`. Implements connection pooling and explicit `UniqueConstraint` on snapshots for production resilience.

### 2. Feature Engineering Logic (`src/features/`)
Features are derived to capture institutional positioning:
-   **Black-Scholes Engine**: Custom iterative IV solver (Newton-Raphson/Bisection fallback) and analytical Greek formulas. Generates **Delta, Gamma, Theta, Vega, and IV** on-the-fly for historical data.
-   **Dealer Positioning**: Computes **Net GEX (Gamma Exposure)**. Identifies "Gamma Flip" zones where market volatility shifts from mean-reverting to trending.
-   **Order Flow Analysis**: Computes **OI Velocity**, **ATM Imbalance**, and **Trap Scores** to identify aggressive delta positioning by market makers.

### 3. Machine Learning Strategy (`src/models/`)
-   **Multi-Regime Ensemble**: Specialized LightGBM models per regime (`RANGE_BOUND`, `TRENDING_UP`, etc.).
-   **Training Alignment**: Optimized for **1-day horizons** (1440 min) to bridge EOD snapshots with intraday follow-through.
-   **Stability Audit**: Strict verification via **TimeSeriesSplit** requiring Mean AUC > 0.52 and Std Dev < 0.15 across folds to prevent over-fitting.
-   **Calibration**: Uses **Isotonic Regression** to transform raw log-loss outputs into calibrated probabilities for geometric sizing.

### 4. Regime-Adaptive Strategy (`src/strategy/`)
-   **RegimeClassifier**: Decision-tree logic that segments the market based on ATR, Trend Strength, and IV Percentiles.
-   **RiskManager**: Implements **Fractional Kelly Criterion**.
-   **TelegramBot**: Real-time signal transmission with structured markdown alerts.

## 🚀 Quick Start & Verification

### 1. Environment Setup
```bash
# Clone and enter workspace
git clone <repo-url> && cd antigravity

# Install dependencies
pip3 install -r requirements.txt

# Configure environment
cp .env.example .env  # Populate with Groww JWT and DB URL
```

### 2. Physical Verification (Black-Scholes & Data)
Before training, verify the mathematical integrity of the engine:
```bash
# 1. Ingest a small sample of Bhavcopy data
./.venv/bin/python3 -m src.data.nse_bhavcopy_loader --months 1 --symbol NIFTY

# 2. Run the BS Engine verification script
./.venv/bin/python3 verify_bs_engine.py
```
*Verify that IV and Delta values are physically consistent (e.g., ATM Call Delta ≈ 0.5).*

### 3. Historical Ingestion (6 Months)
To provide enough trading days for the Stability Audit:
```bash
./.venv/bin/python3 -m src.data.nse_bhavcopy_loader --months 6 --symbol NIFTY
```

### 4. Training the Ensemble
Execute the regime-specific training pipeline:
```bash
./.venv/bin/python3 -m src.train_model
```
*Successful completion will save models in the `models/` directory if they pass the Stability Audit (Mean AUC > 0.52, Std < 0.15).*

### 5. Live Execution & Scanning
```bash
python3 -m src.main
```

---

## 🚀 Deployment & Operations
### Production Workflow
1.  **Configure**: Populate `.env` with API credentials and PostgreSQL details.
2.  **Ingest**: Run `./.venv/bin/python3 -m src.data.nse_bhavcopy_loader --months 6 --symbol NIFTY`
3.  **Train**: Run `./.venv/bin/python3 -m src.train_model`
4.  **Trade**: Run `./.venv/bin/python3 -m src.main`

### Context for LLM Agents
-   **Lints**: The project uses strict type hints. New features in `src/features/` must be vectorized for performance.
-   **Database Integrity**: The `option_chain_snapshots` table uses a unique constraint on `(timestamp, expiry, strike, option_type, symbol)`. Use `.on_conflict_do_nothing()` for inserts.
-   **Data Safety**: The feature pipeline enforces `lag_safety=True` to prevent look-ahead bias during training.
