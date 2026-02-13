import asyncio
import aiohttp
from src.data.groww_client import GrowwClient
import json

async def trace_option_chain():
    client = GrowwClient()
    print("Fetching option chain for NIFTY...")
    raw_data = await client.get_option_chain("NIFTY")
    
    if not raw_data:
        print("Failed to fetch data.")
        return

    underlying = raw_data.get('company', {}).get('ltp')
    print(f"Underlying Price from company metadata: {underlying}")
    
    contracts = raw_data.get('optionChain', {}).get('optionContracts', [])
    print(f"Number of contracts: {len(contracts)}")
    
    if contracts:
        sample = contracts[0]
        ce = sample.get('ce', {})
        pe = sample.get('pe', {})
        print(f"Sample Strike {sample.get('strikePrice')}: CE GrowwID={ce.get('growwContractId')}, PE GrowwID={pe.get('growwContractId')}")

    snapshots = client.parse_option_chain(raw_data)
    if snapshots:
        print(f"Parsed {len(snapshots)} snapshots.")
        print(f"First snapshot underlying_price: {snapshots[0].get('underlying_price')}")
        print(f"First snapshot LTP: {snapshots[0].get('ltp')}")
    else:
        print("No snapshots parsed.")

if __name__ == "__main__":
    asyncio.run(trace_option_chain())
