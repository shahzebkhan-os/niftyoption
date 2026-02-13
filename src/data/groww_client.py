import os
import logging
import asyncio
import aiohttp
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

class GrowwClient:
    def __init__(self, jwt_token=None, secret=None):
        self.jwt_token = jwt_token or os.getenv('GROWW_JWT_TOKEN')
        self.secret = secret or os.getenv('GROWW_SECRET')
        self.base_url = "https://groww.in"
        self.logger = logging.getLogger(__name__)
        self.headers = {
            "Authorization": f"Bearer {self.jwt_token}",
            "X-App-Id": "growwWeb",
            "x-platform": "web",
            "x-device-type": "desktop",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        self.symbol_map = {
            "BANKNIFTY": {"chain": "NIFTY-BANK", "spot": "BANKNIFTY"},
            "NIFTYBANK": {"chain": "NIFTY-BANK", "spot": "BANKNIFTY"},
            "NIFTY": {"chain": "NIFTY", "spot": "NIFTY"}
        }

    def _resolve_symbol(self, symbol: str, context: str = "chain") -> str:
        symbol_upper = symbol.upper()
        if symbol_upper in self.symbol_map:
            return self.symbol_map[symbol_upper].get(context, symbol)
        return symbol

    async def get_historical_candles(self, symbol="NIFTY", start_time=None, end_time=None, interval=1):
        """
        Fetches historical candles using the internal charting_service endpoint.
        start_time/end_time: datetime objects or ISO strings.
        interval: 1, 5, 15, 30, 60, 1440.
        """
        symbol = self._resolve_symbol(symbol, context="spot")
        # Convert times to milliseconds
        if isinstance(start_time, str):
            start_ts = int(datetime.fromisoformat(start_time.replace("Z", "+00:00")).timestamp() * 1000)
        elif isinstance(start_time, datetime):
            start_ts = int(start_time.timestamp() * 1000)
        else:
            start_ts = int((datetime.now() - timedelta(days=30)).timestamp() * 1000)

        if isinstance(end_time, str):
            end_ts = int(datetime.fromisoformat(end_time.replace("Z", "+00:00")).timestamp() * 1000)
        elif isinstance(end_time, datetime):
            end_ts = int(end_time.timestamp() * 1000)
        else:
            end_ts = int(datetime.now().timestamp() * 1000)

        # Map symbol segments
        # Indices are usually NSE/CASH
        exchange = "NSE"
        segment = "CASH"
        
        endpoint = f"/v1/api/charting_service/v2/chart/exchange/{exchange}/segment/{segment}/{symbol}"
        params = {
            "startTimeInMillis": start_ts,
            "endTimeInMillis": end_ts,
            "intervalInMinutes": interval
        }

        async with aiohttp.ClientSession() as session:
            return await self._get(session, endpoint, params)

    async def _post(self, session, endpoint, data=None):
        url = f"{self.base_url}{endpoint}"
        try:
            async with session.post(url, headers=self.headers, json=data) as response:
                if response.status in [200, 206]:
                    return await response.json()
                elif response.status == 401:
                    self.logger.warning("Groww API Unauthorized (401). Your JWT token might be expired. Please update GROWW_JWT_TOKEN in .env")
                    return None
                else:
                    self.logger.error(f"Error posting to {url}: {response.status}")
                    text = await response.text()
                    self.logger.error(f"Response: {text}")
                    return None
        except Exception as e:
            self.logger.error(f"Exception in POST request: {e}")
            return None

    async def _get(self, session, endpoint, params=None):
        url = f"{self.base_url}{endpoint}"
        try:
            async with session.get(url, headers=self.headers, params=params) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 429:
                    self.logger.warning("Rate limit hit. Sleeping...")
                    await asyncio.sleep(1)
                    return await self._get(session, endpoint, params)
                elif response.status == 401:
                    self.logger.warning("Groww API Unauthorized (401). Your JWT token might be expired. Please update GROWW_JWT_TOKEN in .env")
                    return None
                else:
                    self.logger.error(f"Error fetching {url}: {response.status}")
                    text = await response.text()
                    self.logger.error(f"Response: {text}")
                    return None
        except Exception as e:
            self.logger.error(f"Exception in GET request: {e}")
            return None

    async def get_available_expiries(self, symbol="NIFTY"):
        """
        Fetches expiry dates bundled in the pro-option-chain endpoint.
        """
        symbol = self._resolve_symbol(symbol, context="chain")
        endpoint = f"/v1/pro-option-chain/{symbol.lower()}"
        params = {"responseStructure": "LIST"}
        async with aiohttp.ClientSession() as session:
            data = await self._get(session, endpoint, params)
            if data and isinstance(data, dict):
                # Bundle found in aggregatedDetails
                agg = data.get('optionChain', {}).get('aggregatedDetails', {})
                return agg.get('expiryDates', [])
            
        return []

    async def get_indices_ltp(self, symbol="NIFTY"):
        """
        Fetches the underlying spot price for an index using the accord_points endpoint.
        Uses a separate session without the auth JWT to avoid 401s on public routes.
        """
        symbol = self._resolve_symbol(symbol, context="spot")
        endpoint = f"/v1/api/stocks_data/v1/accord_points/exchange/NSE/segment/CASH/latest_indices_ohlc/{symbol}"
        url = f"{self.base_url}{endpoint}"
        
        # Use minimal headers for public endpoint to avoid 401 if JWT is expired
        public_headers = {
            'X-App-Id': 'growwWeb',
            'x-platform': 'web',
            'x-device-type': 'desktop'
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, headers=public_headers) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        self.logger.warning(f"Public spot fetch failed with status {response.status}")
                        # Fallback to standard internal fetch
                        return await self._get(session, endpoint)
            except Exception as e:
                self.logger.error(f"Error in get_indices_ltp: {e}")
                return None

    async def get_live_prices(self, contract_ids):
        """
        Fetches live LTP and OI for a batch of contract IDs (batched to 50 per request).
        """
        if not contract_ids: return {}
        endpoint = "/v1/api/stocks_fo_data/v1/tr_live_prices/exchange/NSE/segment/FNO/latest_prices_batch"
        
        all_live_data = {}
        batch_size = 50
        
        async with aiohttp.ClientSession() as session:
            for i in range(0, len(contract_ids), batch_size):
                batch = contract_ids[i:i + batch_size]
                data = await self._post(session, endpoint, batch)
                if data and isinstance(data, dict):
                    all_live_data.update(data)
                
        return all_live_data

    async def get_option_chain(self, symbol="NIFTY", expiry=None):
        """
        Fetches the option chain and supplements it with live prices.
        """
        chain_symbol = self._resolve_symbol(symbol, context="chain")
        if not expiry:
            expiries = await self.get_available_expiries(symbol)
            expiry = expiries[0] if expiries else None
            
        if not expiry: return None

        endpoint = f"/v1/pro-option-chain/{chain_symbol.lower()}"
        params = {"expiryDate": expiry, "responseStructure": "LIST"}

        async with aiohttp.ClientSession() as session:
            # Parallel fetch: Option Chain structure + Spot Price
            # 1. Option Chain
            chain_task = self._get(session, endpoint, params)
            # 2. Spot Price (Reliable source since company.ltp is often null)
            spot_task = self.get_indices_ltp(symbol)
            
            data, spot_data = await asyncio.gather(chain_task, spot_task)
            
            if not data: return None
            
            # Inject Spot Price into the structure for the parser
            if spot_data and isinstance(spot_data, dict):
                if 'company' not in data: data['company'] = {}
                data['company']['ltp'] = spot_data.get('value')

            # Supplement with Live Prices (LTP/OI for all contracts)
            contracts = data.get('optionChain', {}).get('optionContracts', [])
            contract_ids = []
            for c in contracts:
                if c.get('ce', {}).get('growwContractId'): contract_ids.append(c['ce']['growwContractId'])
                if c.get('pe', {}).get('growwContractId'): contract_ids.append(c['pe']['growwContractId'])
            
            if contract_ids and data is not None:
                live_data = await self.get_live_prices(contract_ids)
                # Inject back into data for the parser
                data['_live_prices'] = live_data
                
            return data

    async def fetch_option_chain(self, symbol: str):
        """Wrapper for compatibility with Step 1 structure."""
        return await self.get_option_chain(symbol)

    async def connect(self):
        """Place holder for session management."""
        pass

    async def close(self):
        """Place holder for session management."""
        pass

    def parse_option_chain(self, raw_data, symbol="NIFTY"):
        """
        Parses the production 'pro-option-chain' response with merged live prices.
        """
        if not raw_data or not isinstance(raw_data, dict):
            return []
        
        chain_container = raw_data.get('optionChain', {})
        agg_details = chain_container.get('aggregatedDetails', {})
        current_expiry = agg_details.get('currentExpiry')
        contracts = chain_container.get('optionContracts', [])
        live_prices = raw_data.get('_live_prices') or {}
        
        if not contracts:
            self.logger.warning("No option contracts found in response.")
            return []
            
        # Extract underlying price for denormalization
        underlying_price = raw_data.get('company', {}).get('ltp')
            
        snapshots = []
        timestamp = datetime.utcnow()
        
        for item in contracts:
            raw_strike = item.get('strikePrice', 0)
            strike = raw_strike / 100.0 if raw_strike > 100000 else raw_strike
            
            def extract_data(option, type_):
                if not option: return None
                contract_id = option.get('growwContractId')
                # Merge live data from supplemental batch fetch
                live = live_prices.get(contract_id, option.get('liveData', {}))
                greeks = option.get('greeks', live.get('greeks', {}))
                
                return {
                    'timestamp': timestamp,
                    'expiry': option.get('expiryDate') or current_expiry,
                    'strike': strike,
                    'option_type': type_,
                    'ltp': live.get('ltp', live.get('lastPrice')),
                    'volume': live.get('volume'),
                    'oi': live.get('openInterest', live.get('oi')),
                    'oi_change': live.get('oiDayChange', live.get('changeinOpenInterest')),
                    'bid_price': live.get('bidPrice'),
                    'ask_price': live.get('askPrice'),
                    'iv': live.get('iv', greeks.get('iv')),
                    'delta': greeks.get('delta'),
                    'gamma': greeks.get('gamma'),
                    'vega': greeks.get('vega'),
                    'theta': greeks.get('theta'),
                    'symbol': symbol,
                    'underlying_price': underlying_price
                }

            ce_data = extract_data(item.get('ce'), 'CE')
            if ce_data: snapshots.append(ce_data)
            
            pe_data = extract_data(item.get('pe'), 'PE')
            if pe_data: snapshots.append(pe_data)
                
        return snapshots
