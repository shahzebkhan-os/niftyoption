import os
import requests
import zipfile
import io
import pandas as pd
import logging
import argparse
from datetime import datetime, timedelta
from typing import List, Optional
from sqlalchemy.dialects.postgresql import insert
from tqdm import tqdm
from src.data.database import OptionChainSnapshot, get_engine, get_session

# Configuration
# Working UDiFF pattern confirmed
URL_PATTERNS = [
    "https://nsearchives.nseindia.com/content/fo/BhavCopy_NSE_FO_0_0_0_{date_ymd}_F_0000.csv.zip",
    "https://nsearchives.nseindia.com/content/fo/BhavCopy_F_O_UDIFF_{date_dmy}.zip",
]
TMP_DIR = "/tmp/nse_bhavcopy"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/all-reports-derivatives",
}

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BhavcopyLoader:
    def __init__(self, symbol="NIFTY"):
        self.symbol = symbol.upper()
        self.engine = get_engine()
        self.session = get_session()
        self.session_http = requests.Session()
        self.session_http.headers.update(HEADERS)
        os.makedirs(TMP_DIR, exist_ok=True)

    def _get_trading_days(self, months: int = None, start_date: datetime = None, end_date: datetime = None) -> List[datetime]:
        """Generates a list of potential trading days (Monday to Friday)."""
        if months:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30 * months)
        
        if not start_date or not end_date:
            raise ValueError("Must provide either months or start/end dates.")

        days = []
        current = start_date
        while current <= end_date:
            if current.weekday() < 5:  # Mon-Fri
                days.append(current)
            current += timedelta(days=1)
        return days

    def _download_and_extract(self, date: datetime) -> Optional[pd.DataFrame]:
        """Downloads ZIP, extracts CSV, and returns a DataFrame. Tries multiple URL patterns."""
        dmy = date.strftime("%d%m%Y")
        ymd = date.strftime("%Y%m%d")
        
        for pattern in URL_PATTERNS:
            url = pattern.format(date_dmy=dmy, date_ymd=ymd)
            try:
                # Use a small wait to be polite to NSE
                response = self.session_http.get(url, timeout=15)
                if response.status_code == 200:
                    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                        csv_files = [f for f in z.namelist() if f.endswith('.csv')]
                        if not csv_files:
                            continue
                        with z.open(csv_files[0]) as f:
                            df = pd.read_csv(f)
                            df.columns = [c.strip() for c in df.columns]
                            return df
                elif response.status_code == 404:
                    continue
            except Exception:
                continue
        return None

    def _parse_df(self, df: pd.DataFrame, trading_date: datetime) -> pd.DataFrame:
        """Filters and maps the confirmed UDiFF bhavcopy data to our schema."""
        # Confirmed UDiFF Columns: TradDt, BizDt, Sgmt, Src, FinInstrmTp, TckrSymb, XpryDt, StrkPric, OptnTp, ClsPric, OpnIntrst, ChngInOpnIntrst, TtlTradgVol, UndrlygPric
        
        if 'TradDt' in df.columns:
            symbol_col = 'TckrSymb'
            opt_type_col = 'OptnTp'
            strike_col = 'StrkPric'
            expiry_col = 'XpryDt'
            close_col = 'ClsPric'
            oi_col = 'OpnIntrst'
            qty_col = 'TtlTradgVol'
            chg_oi_col = 'ChngInOpnIntrst'
            underlying_col = 'UndrlygPric'
        else:
            # Fallback for old/other formats
            symbol_col = 'SYMBOL' if 'SYMBOL' in df.columns else 'Symbol'
            opt_type_col = 'OPTION_TYP' if 'OPTION_TYP' in df.columns else 'OptnType'
            strike_col = 'STRIKE_PR' if 'STRIKE_PR' in df.columns else 'StrikePr'
            expiry_col = 'EXPIRY_DT' if 'EXPIRY_DT' in df.columns else 'ExpiryDt'
            close_col = 'CLOSE' if 'CLOSE' in df.columns else 'Close'
            oi_col = 'OPEN_INT' if 'OPEN_INT' in df.columns else 'OI'
            qty_col = 'CONTRACTS' if 'CONTRACTS' in df.columns else 'Qty'
            chg_oi_col = 'CHG_IN_OI' if 'CHG_IN_OI' in df.columns else 'ChgInOI'
            underlying_col = None

        df = df[
            (df[symbol_col].astype(str).str.upper() == self.symbol) & 
            (df[opt_type_col].isin(['CE', 'PE']))
        ].copy()

        if df.empty:
            return pd.DataFrame()

        # 2. Map Columns - Ensure date-only for training stability
        df['expiry'] = pd.to_datetime(df[expiry_col])
        df['timestamp'] = pd.to_datetime(trading_date.date()) # Cast to start of day
        
        mapped_df = pd.DataFrame({
            'timestamp': df['timestamp'],
            'expiry': df['expiry'],
            'strike': df[strike_col].astype(float),
            'option_type': df[opt_type_col],
            'ltp': df[close_col].astype(float),
            'volume': df[qty_col].astype(int),
            'oi': df[oi_col].astype(int),
            'oi_change': df[chg_oi_col].astype(int),
            'symbol': df[symbol_col],
            'underlying_price': df[underlying_col].astype(float) if underlying_col and underlying_col in df.columns else None
        })
        
        return mapped_df

    def _bulk_upsert(self, df: pd.DataFrame, batch_size=500):
        """Performs batched bulk upsert to avoid parameter limits."""
        if df.empty:
            return

        records = df.to_dict(orient='records')
        
        for i in range(0, len(records), batch_size):
            batch = records[i : i + batch_size]
            stmt = insert(OptionChainSnapshot.__table__).values(batch)
            stmt = stmt.on_conflict_do_nothing(
                index_elements=['timestamp', 'expiry', 'strike', 'option_type', 'symbol']
            )
            
            try:
                self.session.execute(stmt)
                self.session.commit()
            except Exception as e:
                self.session.rollback()
                logger.error(f"Failed to upsert batch: {e}")

    def run(self, months: int = None, start_date: datetime = None, end_date: datetime = None):
        """Main entry point for ingestion."""
        days = self._get_trading_days(months, start_date, end_date)
        logger.info(f"Starting ingestion for {len(days)} potential trading days...")
        
        total_inserted = 0
        all_dates_processed = []

        for date in tqdm(days, desc="Ingesting Bhavcopies"):
            raw_df = self._download_and_extract(date)
            if raw_df is None:
                # logger.debug(f"No data for {date.date()} (Holiday or skip)")
                continue
                
            parsed_df = self._parse_df(raw_df, date)
            if not parsed_df.empty:
                self._bulk_upsert(parsed_df)
                total_inserted += len(parsed_df)
                all_dates_processed.append(date.date())
                # logger.info(f"Inserted {len(parsed_df)} rows for {date.date()}")

        if all_dates_processed:
            logger.info("========================================================")
            logger.info(f"LOAD COMPLETE: {self.symbol}")
            logger.info(f"Total rows handled: {total_inserted}")
            logger.info(f"Date range: {min(all_dates_processed)} to {max(all_dates_processed)}")
            logger.info("========================================================")
        else:
            logger.warning("No data was ingested. Check date range or network connectivity.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NSE FO Bhavcopy Loader")
    parser.add_argument("--months", type=int, help="Number of months to look back")
    parser.add_argument("--symbol", type=str, default="NIFTY", help="Symbol to parse (NIFTY/BANKNIFTY)")
    parser.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD)")
    
    args = parser.parse_args()
    
    start_dt, end_dt = None, None
    if args.start_date:
        start_dt = datetime.strptime(args.start_date, "%Y-%m-%d")
    if args.end_date:
        end_dt = datetime.strptime(args.end_date, "%Y-%m-%d")
        
    loader = BhavcopyLoader(symbol=args.symbol)
    loader.run(months=args.months, start_date=start_dt, end_date=end_dt)
