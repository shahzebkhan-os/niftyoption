# Intraday Options Intelligence Engine

## 🧠 System Thesis
This project is a production-grade, capital-preserving intelligence engine designed to predict significant intraday moves in equity indices (NIFTY/BANKNIFTY) using high-frequency options chain data. It prioritizes **probabilistic edge** over high-frequency turnover, utilizing a regime-adaptive ensemble approach.

---

## 🏗 Architectural Blueprint (AI-Context)
The system is built as a modular, asynchronous pipeline divided into five distinct layers.

### 1. Data Ingestion Layer (`src/data/`)
-   **GrowwClient**: Asynchronous `aiohttp` client. Handles fragmented JSON responses, session persistence, and rate-limiting backoff.
-   **MarketDataFetcher**: Continuous polling loop that serializes full options chain snapshots (multiple strikes/expiries).
-   **Database**: PostgreSQL storage utilizing `SQLAlchemy` with `psycopg2-binary`. Implements connection pooling and `tenacity` retry logic for production resilience.

### 2. Feature Engineering Logic (`src/features/`)
Features are derived from raw chain data to capture institutional positioning:
-   **Dealer Positioning (Greeks)**: Computes **Net GEX (Gamma Exposure)** by aggregating Open Interest and Gamma across all strikes. Identifies "Gamma Flip" zones where market volatility shifts from mean-reverting to trending.
-   **Volatility Intelligence**: Tracks **IV Percentile** and **IV Velocity** (rate of change of implied volatility) to detect front-running of news or systemic volatility expansion.
-   **Order Flow Analysis**: Computes **OI Velocity** and **ATM Imbalance** to identify aggressive delta positioning by market makers.

### 3. Machine Learning Strategy (`src/models/`)
-   **Target Definition**: A binary classification target: $P(|\text{Move}| > X \text{ points in } T \text{ minutes})$.
-   **Model**: LightGBM/XGBoost classifier.
-   **Validation**: **Walk-Forward TimeSeriesSplit** ensures no data leakage and validates the model's performance on rolling market regimes.
-   **Calibration**: Uses **Platt Scaling / Isotonic Regression** to transform raw log-loss outputs into calibrated probabilities (0.0 - 1.0) suitable for Kelly sizing.

### 4. Regime-Adaptive Strategy (`src/strategy/`)
-   **RegimeClassifier**: A decision-tree based classifier that segments the market into `TRENDING_UP`, `TRENDING_DOWN`, `RANGE_BOUND`, and `HIGH_IV`.
-   **RiskManager**: Implements **Fractional Kelly Criterion**. Position size $= f \cdot (\text{Prob} - (1-\text{Prob}) / \text{RR})$.
-   **TelegramBot**: Structured, markdown-formatted alerting system for real-time signal transmission.

### 5. Institutional Backtesting Engine v2 (`src/backtest.py`)
A custom **Event-Driven Simulator** that eliminates "Look-Ahead Bias":
-   **Bar-by-Bar Replay**: Iterates through historical price action without peeking at future rows.
-   **Execution Modeling**: Includes **Slippage** (bid-ask spread impact) and **Brokerage Fees**.
-   **Gap Realism**: If a stop-loss is triggered via a gap-down, the engine fills at the `Open` price rather than the theoretical `Stop` price.
-   **State Machine**: Tracks `cooldowns`, `consecutive losses`, and `open_position` objects.

---

## 🚀 Deployment & Operations
### Production Workflow
1.  **Configure**: Populate `.env` with API credentials and PostgreSQL details.
2.  **Verify**: Run `python3 -m src.utils.verify_setup` to test connectivity.
3.  **Deploy**: Use `docker-compose up --build -d` for a containerized stack.

### Context for LLM Agents
If you are an AI model tasked with modifying this codebase:
-   **Lints**: The project uses strict type hints where possible. Ensure new features in `src/features/` return dictionaries for easy consumption by the `RegimeClassifier`.
-   **Async Context**: Most of `src/data/` and `src/main.py` is asynchronous. Do not block the event loop with heavy CPU-bound tasks; use `ProcessPoolExecutor` if necessary.
-   **Data Leakage**: Never use `.shift()` with negative periods in the backtester; it will introduce look-ahead bias that voids performance metrics.
