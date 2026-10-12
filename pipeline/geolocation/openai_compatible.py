"""Shared OpenAI-compatible vision client (OpenRouter, Vercel AI Gateway, etc.)."""

import base64
import json
import logging
import mimetypes
import os
import re
import time
from collections import Counter
from typing import Optional, Sequence

import requests

logger = logging.getLogger(__name__)


def _encode_image(image_path: str) -> tuple[str, str]:
    mime, _ = mimetypes.guess_type(image_path)
    if mime is None:
        mime = "image/png"
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("ascii")
    return mime, b64


def _extract_json(text: str) -> dict:
    text = (text or "").strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def chat_json(
    *,
    url: str,
    api_key: str,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    image_path: str,
    timeout: float = 120.0,
    max_tokens: int = 512,
    extra_headers: Optional[dict] = None,
) -> dict:
    mime, b64 = _encode_image(image_path)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if extra_headers:
        headers.update(extra_headers)

    payload = {
        "model": model_name,
        "max_tokens": max_tokens,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                    },
                ],
            },
        ],
        "response_format": {"type": "json_object"},
    }

    response = requests.post(url, headers=headers, json=payload, timeout=timeout)
    if response.status_code >= 400:
        logger.error("Vision API error %s: %s", response.status_code, response.text[:500])
        response.raise_for_status()

    data = response.json()
    content = data["choices"][0]["message"]["content"]
    if isinstance(content, list):
        text_parts = [part.get("text", "") for part in content if isinstance(part, dict)]
        content = "\n".join(text_parts)
    return _extract_json(content)


class CompatibleGeolocator:
    """Image-based geolocation via any OpenAI-compatible vision endpoint."""

    def __init__(
        self,
        *,
        url: str,
        api_key: str,
        model_name: str,
        exclude_keywords: Optional[Sequence[str]] = None,
        delay_seconds: float = 2.0,
        max_tokens: int = 512,
        extra_headers: Optional[dict] = None,
        provider_name: str = "OpenAI-compatible",
    ):
        if not api_key:
            raise RuntimeError(f"{provider_name} API key is not set.")
        self.url = url
        self.api_key = api_key
        self.model_name = model_name
        self.delay_seconds = delay_seconds
        self.max_tokens = max_tokens
        self.extra_headers = extra_headers
        self.provider_name = provider_name
        self.exclude_keywords = exclude_keywords or ["top_view"]
        self.system_prompt = (
            "You are a highly specialized geographic AI expert. "
            "Your primary goal is to identify specific, non-generic landmarks in the provided image "
            "(e.g., street signs, unique architectural details, specific shop names, utility poles, mountain peaks, road markings). "
            "Use these unique features to cross-reference with satellite and street-view data "
            "to determine the coordinates with the highest possible precision. "
            "Return only JSON."
        )
        self.user_prompt = (
            "ANALYZE THE IMAGE FOR UNIQUE DETAILS. Provide the GPS coordinates (latitude and longitude) "
            "where the photo was taken with the highest possible precision (ideally within 10 meters). "
            "Only if the image is entirely devoid of unique landmarks, you can lower the accuracy. "
            "Return strictly in JSON format like this: "
            '{"latitude": <value>, "longitude": <value>}.'
        )

    def geolocate_image(self, image_path: str) -> Optional[dict]:
        try:
            coords = chat_json(
                url=self.url,
                api_key=self.api_key,
                model_name=self.model_name,
                system_prompt=self.system_prompt,
                user_prompt=self.user_prompt,
                image_path=image_path,
                max_tokens=self.max_tokens,
                extra_headers=self.extra_headers,
            )
            latitude = coords.get("latitude")
            longitude = coords.get("longitude")
            if latitude is not None and longitude is not None:
                return {"latitude": float(latitude), "longitude": float(longitude)}
            return None
        except Exception as e:
            logger.error("%s geolocation failed for %s: %s", self.provider_name, image_path, e)
            return None

    def run_pipeline(self, folder_path: str):
        if not os.path.isdir(folder_path):
            return None, None

        image_files = sorted(
            [
                f
                for f in os.listdir(folder_path)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
                and not any(keyword in f.lower() for keyword in self.exclude_keywords)
            ]
        )
        if not image_files:
            return None, None

        all_predictions = []
        for i, filename in enumerate(image_files):
            if i > 0:
                time.sleep(self.delay_seconds)
            result = self.geolocate_image(os.path.join(folder_path, filename))
            if result:
                all_predictions.append((result["latitude"], result["longitude"]))

        counter = Counter(all_predictions)
        most_common = counter.most_common(1)[0][0] if counter else None
        return most_common, counter


class CompatibleDimensionEstimator:
    """Estimate scene width (meters) via any OpenAI-compatible vision endpoint."""

    def __init__(
        self,
        *,
        url: str,
        api_key: str,
        model_name: str,
        max_tokens: int = 512,
        extra_headers: Optional[dict] = None,
        provider_name: str = "OpenAI-compatible",
    ):
        if not api_key:
            raise RuntimeError(f"{provider_name} API key is not set.")
        self.url = url
        self.api_key = api_key
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.extra_headers = extra_headers
        self.provider_name = provider_name
        self.system_prompt = (
            "You are an expert in spatial analysis and image photogrammetry. "
            "Your task is to analyze images to estimate the physical scale and dimensions of the scene. "
            "Use visual cues such as objects with standard sizes (cars, road lanes, doors, people), "
            "perspective, and depth of field to calculate the size of the area shown. "
            "Return only JSON. Never return null for numeric fields; provide a plausible positive estimate if unsure."
        )
        self.user_prompt = (
            "Analyze the provided image to estimate the physical width of the captured area in meters. "
            "Focus specifically on the width of the visible scene. "
            'Return JSON only like: {"dimension_meters": 25.5}'
        )

    def estimate_dimension(self, image_path: str) -> Optional[float]:
        try:
            result = chat_json(
                url=self.url,
                api_key=self.api_key,
                model_name=self.model_name,
                system_prompt=self.system_prompt,
                user_prompt=self.user_prompt,
                image_path=image_path,
                max_tokens=self.max_tokens,
                extra_headers=self.extra_headers,
            )
            val = result.get("dimension_meters")
            return float(val) if val is not None else None
        except Exception as e:
            logger.error("Error estimating dimension via %s: %s", self.provider_name, e)
            return None
