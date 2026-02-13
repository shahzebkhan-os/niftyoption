import asyncio
import logging
from datetime import datetime, timedelta
from src.data.groww_client import GrowwClient
from src.data.database import get_session, OptionChainSnapshot

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def load_historical_data(symbol="NIFTY", months=6):
    """
    Orchestrates the loading of historical spot data.
    """
    client = GrowwClient()
    session = get_session()
    
    end_time = datetime.now()
    start_time = end_time - timedelta(days=30 * months)
    
    # Format for Groww API: yyyy-MM-dd HH:mm:ss
    start_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
    end_str = end_time.strftime("%Y-%m-%d %H:%M:%S")
    
    logger.info(f"Loading historical data for {symbol} from {start_str} to {end_str}")
    
    try:
        # Fetch 1-hour candles for efficiency over long history
        # (1-minute is often restricted to recent days/weeks)
        response = await client.get_historical_candles(
            symbol=symbol,
            start_time=start_str,
            end_time=end_str,
            interval=60 # 60 minutes
        )
        
        if not response or 'candles' not in response:
            logger.error(f"Failed to fetch historical data: {response}")
            return

        candles = response['candles']
        logger.info(f"Fetched {len(candles)} candles. Injecting into database...")
        
        for candle in candles:
            # candle structure from docs: [timestamp, open, high, low, close, volume]
            ts_epoch = candle[0]
            close_price = candle[4]
            ts_dt = datetime.fromtimestamp(ts_epoch)
            
            snapshot = OptionChainSnapshot(
                timestamp=ts_dt,
                expiry=ts_dt, # Dummy
                strike=0.0,
                option_type='SP', # 'SP' for Spot (fits in String(2))
                underlying_price=close_price,
                symbol=symbol,
                ltp=close_price,
                oi=0,
                iv=0.0
            )
            session.add(snapshot)
            
        session.commit()
        logger.info(f"Successfully loaded {len(candles)} historical spot records.")

    except Exception as e:
        logger.error(f"Error loading historical data: {e}")
        session.rollback()
    finally:
        session.close()

if __name__ == "__main__":
    asyncio.run(load_historical_data())
