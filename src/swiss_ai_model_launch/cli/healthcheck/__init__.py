from enum import Enum

import httpx


class ModelHealth(Enum):
    WAITING = "WAITING"
    HEALTHY = "HEALTHY"
    NOT_RESPONDING = "NOT_RESPONDING"


_HEALTH_CHECK_URL = "https://api.swissai.cscs.ch/v1/chat/completions"


async def check_model_health(served_model_name: str, api_key: str) -> ModelHealth:
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                _HEALTH_CHECK_URL,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                },
                json={
                    "model": served_model_name,
                    "messages": [{"role": "user", "content": "Say hello."}],
                    "stream": False,
                },
                timeout=30,
            )
        return (
            ModelHealth.HEALTHY if response.is_success else ModelHealth.NOT_RESPONDING
        )
    except httpx.TransportError:
        return ModelHealth.NOT_RESPONDING
