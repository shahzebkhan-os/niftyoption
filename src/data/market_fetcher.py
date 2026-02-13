import asyncio
import logging
import json
from datetime import datetime
from src.data.groww_client import GrowwClient
from src.data.database import get_session, OptionChainSnapshot, RawDataLog

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MarketDataFetcher:
    def __init__(self, symbol="NIFTY", interval=60):
        self.symbol = symbol
        self.interval = interval
        self.client = GrowwClient()
        self.running = False

    async def fetch_and_store(self):
        """
        Fetches option chain and underlying data, then stores in DB.
        """
        session = get_session()
        try:
            # 1. Fetch Option Chain
            logger.info(f"Fetching option chain for {self.symbol}...")
            raw_chain = await self.client.get_option_chain(self.symbol)
            
            if not raw_chain:
                logger.warning("No option chain data received.")
                return []

            # 2. Store Raw Data (Audit Trail)
            raw_log = RawDataLog(
                endpoint=f"option_chain/{self.symbol}",
                raw_data=raw_chain
            )
            session.add(raw_log)
            
            # 3. Parse and Enrich Snapshots
            snapshots_data = self.client.parse_option_chain(raw_chain, symbol=self.symbol)
            if not snapshots_data:
                 logger.warning("No snapshots parsed from data.")
            else:
                 import pandas as pd
                 from src.features.volatility import enrich_with_greeks
                 
                 # Convert to DF for enrichment
                 df = pd.DataFrame(snapshots_data)
                 df = enrich_with_greeks(df)
                 
                 # Back to list of dicts for DB storage
                 snapshots_data = df.to_dict('records')
                 
                 sample_price = snapshots_data[0].get('underlying_price')
                 logger.info(f"Parsed & Enriched {len(snapshots_data)} snapshots. Sample Spot: {sample_price}")
            
            for item in snapshots_data:
                # Clean item for SQLAlchemy (remove extra keys like 'T' if present)
                valid_keys = {c.key for c in OptionChainSnapshot.__table__.columns}
                filtered_item = {k: v for k, v in item.items() if k in valid_keys}
                snapshot = OptionChainSnapshot(**filtered_item)
                session.add(snapshot)
            
            session.commit()
            logger.info(f"Stored {len(snapshots_data)} option snapshots.")
            return snapshots_data

        except Exception as e:
            logger.error(f"Error in fetch_and_store: {e}")
            session.rollback()
            return []
        finally:
            session.close()

    async def get_latest_data(self):
        """
        Triggers a fresh fetch and returns a DataFrame.
        """
        import pandas as pd
        data = await self.fetch_and_store()
        if not data:
            return pd.DataFrame()
        return pd.DataFrame(data)

    async def run_loop(self):
        """
        Continuous fetching loop.
        """
        self.running = True
        logger.info(f"Starting Market Data Fetcher for {self.symbol} every {self.interval}s.")
        while self.running:
            start_time = datetime.now()
            await self.fetch_and_store()
            
            elapsed = (datetime.now() - start_time).total_seconds()
            sleep_time = max(0, self.interval - elapsed)
            logger.info(f"Sleeping for {sleep_time:.2f}s...")
            await asyncio.sleep(sleep_time)

    def stop(self):
        self.running = False

if __name__ == "__main__":
    fetcher = MarketDataFetcher(symbol="NIFTY", interval=60)
    try:
        asyncio.run(fetcher.run_loop())
    except KeyboardInterrupt:
        logger.info("Fetcher stopped by user.")
