import asyncio
import aiohttp
from src.data.groww_client import GrowwClient
import json

async def test_injection():
    client = GrowwClient()
    # Mock data structure similar to Groww API
    data = {
        "company": {"symbol": "NIFTY", "liveData": {}},
        "optionChain": {"optionContracts": []}
    }
    spot_data = {"value": 25559.2}
    
    print(f"Before injection: {data.get('company', {}).get('ltp')}")
    
    # Core logic from GrowwClient.get_option_chain
    if spot_data and isinstance(spot_data, dict):
        if 'company' not in data: data['company'] = {}
        data['company']['ltp'] = spot_data.get('value')
        
    print(f"After injection: {data.get('company', {}).get('ltp')}")
    
    # Test with real API fetch
    print("\n--- Real API Test ---")
    data = await client.get_option_chain("NIFTY")
    if data:
        print(f"Injected ltp in 'data[\"company\"][\"ltp\"]': {data.get('company', {}).get('ltp')}")
        snapshots = client.parse_option_chain(data)
        if snapshots:
            print(f"Snapshot underlying_price: {snapshots[0].get('underlying_price')}")
        else:
            print("No snapshots parsed.")

if __name__ == "__main__":
    asyncio.run(test_injection())
