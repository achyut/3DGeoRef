"""OpenRouter-based geolocation and scene dimension estimation utilities."""

import os

from pipeline.geolocation.openai_compatible import (
    CompatibleDimensionEstimator,
    CompatibleGeolocator,
)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "google/gemini-2.5-flash"


def _api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENROUTER_KEY")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Pass --openrouter_api_key or export OPENROUTER_API_KEY."
        )
    return key


class OpenRouterGeolocator(CompatibleGeolocator):
    def __init__(self, exclude_keywords=None, model_name: str = DEFAULT_MODEL, delay_seconds: float = 2.0):
        super().__init__(
            url=OPENROUTER_URL,
            api_key=_api_key(),
            model_name=model_name,
            exclude_keywords=exclude_keywords,
            delay_seconds=delay_seconds,
            extra_headers={
                "HTTP-Referer": "https://github.com/3DOM-FBK/3DGeoRef",
                "X-Title": "3DGeoRef",
            },
            provider_name="OpenRouter",
        )


class OpenRouterDimensionEstimator(CompatibleDimensionEstimator):
    def __init__(self, model_name: str = DEFAULT_MODEL):
        super().__init__(
            url=OPENROUTER_URL,
            api_key=_api_key(),
            model_name=model_name,
            extra_headers={
                "HTTP-Referer": "https://github.com/3DOM-FBK/3DGeoRef",
                "X-Title": "3DGeoRef",
            },
            provider_name="OpenRouter",
        )
