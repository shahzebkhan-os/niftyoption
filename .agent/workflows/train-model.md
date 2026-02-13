---
description: How to train and update the intraday trading models
---

# Training the Trading Model

This workflow describes the steps to load historical data and train the multi-regime ensemble models.

### Prerequisites
- Ensure `GROWW_JWT_TOKEN` is active in your `.env` file.
- Ensure the PostgreSQL database is running and accessible via the `DATABASE_URL` in `.env`.

### Steps

1. **Load Historical Data**
   Fetch the last 6 months of historical NIFTY spot data into the database.
   // turbo
   ```bash
   python3 -m src.data.historical_loader
   ```

2. **Run Training Pipeline**
   Execute the training orchestration which builds features, generates targets, and trains specialized models for each market regime.
   // turbo
   ```bash
   python3 -m src.train_model
   ```

### Verification
- Check the `models/` directory for `.pkl` files (e.g., `TRENDING_UP_model.pkl`).
- Review the logs for "Multi-regime ensemble training cycle complete".
