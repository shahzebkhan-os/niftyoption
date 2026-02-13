import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from src.data.database import get_engine
from src.features.feature_pipeline import build_features
from src.models.target import TargetGenerator
from sqlalchemy import text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_data():
    engine = get_engine()
    # Load last 1 month for speed
    end = datetime.now()
    start = end - timedelta(days=30)
    
    query = f"SELECT * FROM option_chain_snapshots WHERE symbol = 'NIFTY' AND timestamp >= '{start}' AND underlying_price IS NOT NULL ORDER BY timestamp ASC"
    
    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn)
    
    if df.empty:
        print("No data found.")
        return

    print(f"Total rows: {len(df)}")
    
    # Process features
    df_features = build_features(df, lag_safety=True)
    print(f"Features built: {len(df_features)}")
    
    # Distribution of regimes
    print("\nRegime Distribution:")
    print(df_features['regime'].value_counts())
    
    # Targets
    target_gen = TargetGenerator(horizon_minutes=30, threshold_points=50)
    df_features['target'] = target_gen.generate_target(df_features, price_col='underlying_price', time_col='timestamp')
    
    print("\nTarget Distribution (threshold=50):")
    print(df_features['target'].value_counts(dropna=False))
    
    # Check price changes
    ts_price = df_features.groupby('timestamp')['underlying_price'].first().sort_index()
    future_price = ts_price.shift(-30)
    price_change = (future_price - ts_price).abs()
    print("\nPrice Change Stats (30-min horizon):")
    print(price_change.describe())
    
    print(f"\nMax price change: {price_change.max()}")
    print(f"Threshold (50) percentile: {(price_change < 50).mean() * 100:.2f}%")

if __name__ == "__main__":
    debug_data()
