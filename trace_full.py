import asyncio
import aiohttp
from src.data.groww_client import GrowwClient
import json

async def trace_response():
    client = GrowwClient()
    print("--- Testing get_indices_ltp ---")
    spot_data = await client.get_indices_ltp("NIFTY")
    print(f"Spot Data: {json.dumps(spot_data, indent=2)}")

    print("\n--- Testing get_option_chain ---")
    raw_data = await client.get_option_chain("NIFTY")
    
    if not raw_data:
        print("Failed to fetch option chain.")
        return

    print(f"Top-level keys: {list(raw_data.keys())}")
    if 'company' in raw_data:
        print(f"Company keys: {list(raw_data['company'].keys())}")
        print(f"Company LTP: {raw_data['company'].get('ltp')}")
    
    chain_container = raw_data.get('optionChain', {})
    print(f"OptionChain keys: {list(chain_container.keys())}")
    
    contracts = chain_container.get('optionContracts', [])
    if contracts:
        print(f"\n--- Sample Contract Structure (First) ---")
        sample = contracts[0]
        print(json.dumps(sample, indent=2))
        
        ce = sample.get('ce', {})
        if ce:
            cid = ce.get('growwContractId')
            print(f"\nCE Contract ID: {cid}")
            live_batch = raw_data.get('_live_prices', {})
            print(f"Batch Live Data for {cid}: {json.dumps(live_batch.get(cid), indent=2)}")

if __name__ == "__main__":
    asyncio.run(trace_response())
