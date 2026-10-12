"""Vercel AI Gateway geolocation and scene dimension estimation utilities."""

import os

from pipeline.geolocation.openai_compatible import (
    CompatibleDimensionEstimator,
    CompatibleGeolocator,
)

VERCEL_GATEWAY_URL = "https://ai-gateway.vercel.sh/v1/chat/completions"
# Verified working with vision + JSON on this gateway.
DEFAULT_MODEL = "openai/gpt-4o-mini"


def _api_key() -> str:
    key = (
        os.environ.get("AI_GATEWAY_API_KEY")
        or os.environ.get("VERCEL_AI_GATEWAY_API_KEY")
        or os.environ.get("VERCEL_OIDC_TOKEN")
    )
    if not key:
        raise RuntimeError(
            "AI_GATEWAY_API_KEY is not set. Pass --ai_gateway_api_key or export AI_GATEWAY_API_KEY."
        )
    return key


class VercelGatewayGeolocator(CompatibleGeolocator):
    def __init__(self, exclude_keywords=None, model_name: str = DEFAULT_MODEL, delay_seconds: float = 2.0):
        super().__init__(
            url=VERCEL_GATEWAY_URL,
            api_key=_api_key(),
            model_name=model_name,
            exclude_keywords=exclude_keywords,
            delay_seconds=delay_seconds,
            provider_name="Vercel AI Gateway",
        )


class VercelGatewayDimensionEstimator(CompatibleDimensionEstimator):
    def __init__(self, model_name: str = DEFAULT_MODEL):
        super().__init__(
            url=VERCEL_GATEWAY_URL,
            api_key=_api_key(),
            model_name=model_name,
            provider_name="Vercel AI Gateway",
        )
