import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from src.data.groww_client import GrowwClient

@pytest.mark.asyncio
async def test_groww_client_retry_logic():
    """Verify that GrowwClient retries on 429 Rate Limit."""
    client = GrowwClient(jwt_token="fake_token", secret="fake_secret")
    
    # Mock response objects
    mock_resp_429 = AsyncMock()
    mock_resp_429.status = 429
    
    mock_resp_200 = AsyncMock()
    mock_resp_200.status = 200
    mock_resp_200.json = AsyncMock(return_value={"data": "success"})
    
    # We need to mock aiohttp.ClientSession.get
    with patch("aiohttp.ClientSession.get") as mock_get:
        # First call returns 429, second returns 200
        mock_get.return_value.__aenter__.side_effect = [mock_resp_429, mock_resp_200]
        
        # Patch sleep to avoid waiting in tests
        with patch("asyncio.sleep", return_value=None):
            async with await client._connect_session_mock() as session:
                result = await client._get(session, "/test-endpoint")
                
                assert result == {"data": "success"}
                assert mock_get.call_count == 2

@pytest.mark.asyncio
async def test_groww_client_unauthorized():
    """Verify handling of 401 Unauthorized."""
    client = GrowwClient(jwt_token="expired_token", secret="fake_secret")
    
    mock_resp_401 = AsyncMock()
    mock_resp_401.status = 401
    
    with patch("aiohttp.ClientSession.get") as mock_get:
        mock_get.return_value.__aenter__.return_value = mock_resp_401
        
        async with await client._connect_session_mock() as session:
            result = await client._get(session, "/test-endpoint")
            assert result is None

# Helper to provide a session for testing
async def _connect_session_mock(self):
    import aiohttp
    return aiohttp.ClientSession()

# Inject helper into GrowwClient for tests
GrowwClient._connect_session_mock = _connect_session_mock
