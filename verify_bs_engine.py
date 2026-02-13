import pandas as pd
import logging
from datetime import datetime
from src.data.database import get_session, OptionChainSnapshot
from src.features.volatility import enrich_with_greeks
from sqlalchemy import text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_engine():
    session = get_session()
    
    # Fetch a mix of records (CE, PE, SP) for NIFTY
    # Focus on records that need greeks or have them to verify
    try:
        query = session.query(OptionChainSnapshot).filter(
            OptionChainSnapshot.symbol == 'NIFTY',
            OptionChainSnapshot.underlying_price.isnot(None)
        ).order_by(OptionChainSnapshot.timestamp.desc()).limit(10).all()
        
        if not query:
            print("No data found in database for NIFTY. Please run ingest first.")
            return

        # Convert to DataFrame
        data = []
        for r in query:
            data.append({
                'id': r.id,
                'timestamp': r.timestamp,
                'expiry': r.expiry,
                'strike': r.strike,
                'option_type': r.option_type,
                'underlying_price': r.underlying_price,
                'ltp': r.ltp,
                'iv': r.iv,
                'delta': r.delta,
                'gamma': r.gamma,
                'theta': r.theta,
                'vega': r.vega
            })
        
        df = pd.DataFrame(data)
        
        print("\n=== DATA BEFORE ENRICHMENT ===")
        print(df[['option_type', 'strike', 'underlying_price', 'ltp', 'delta', 'iv']].head())
        
        # Recalculate Greeks
        enriched_df = enrich_with_greeks(df)
        
        print("\n=== DATA AFTER ENRICHMENT ===")
        print(enriched_df[['option_type', 'strike', 'underlying_price', 'ltp', 'delta', 'iv']].head())
        
        # Check if Greeks are populated (except for SP)
        options = enriched_df[enriched_df['option_type'].isin(['CE', 'PE'])]
        if not options.empty:
            avg_delta = options['delta'].abs().mean()
            print(f"\nAverage Absolute Delta for Options: {avg_delta:.4f}")
            if avg_delta > 0:
                print("✅ Greeks are being calculated successfully.")
            else:
                print("❌ Greeks are still zero. Check BS formulas or input parameters (T, r, sigma).")
        
        # Spot check
        spot = enriched_df[enriched_df['option_type'] == 'SP']
        if not spot.empty:
            print(f"\nSpot Record Delta: {spot.iloc[0]['delta']} (Expected: 0)")
            
    except Exception as e:
        logger.error(f"Verification failed: {e}")
    finally:
        session.close()

if __name__ == "__main__":
    verify_engine()
