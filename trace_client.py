import asyncio
import aiohttp
from src.data.groww_client import GrowwClient

async def trace_calls():
    client = GrowwClient()
    print(f"Tracing get_available_expiries...")
    expiries = await client.get_available_expiries("NIFTY")
    print(f"Result: {expiries}")

if __name__ == "__main__":
    asyncio.run(trace_calls())
