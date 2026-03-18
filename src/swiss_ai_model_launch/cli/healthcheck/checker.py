import httpx

from swiss_ai_model_launch.cli.healthcheck.model_health import ModelHealth

_HEALTH_CHECK_URL = "https://api.swissai.svc.cscs.ch/v1/chat/completions"
_MESSAGE = {"role": "user", "content": "Say hello."}
_TIMEOUT_SECONDS = 60


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
                    "messages": [_MESSAGE],
                    "stream": False,
                },
                timeout=_TIMEOUT_SECONDS,
            )
        return ModelHealth.HEALTHY if response.is_success else ModelHealth.ERROR
    except (httpx.TransportError, httpx.TimeoutException):
        return ModelHealth.NOT_RESPONDING
