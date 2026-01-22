"""Configuration loader with .env and Convex fallback support."""

import os
from typing import Optional
import httpx


def get_openrouter_api_key() -> Optional[str]:
    """
    Get OpenRouter API key with fallback:
    1. Check .env (OPENROUTER_API_KEY environment variable)
    2. If not found, fetch from Convex settings
    """
    # First, check environment variable
    env_key = os.environ.get("OPENROUTER_API_KEY")
    if env_key:
        return env_key

    # Fallback to Convex
    convex_key = _get_key_from_convex("openrouterApiKey")
    if convex_key:
        return convex_key

    return None


def get_tavily_api_key() -> Optional[str]:
    """
    Get Tavily API key with fallback:
    1. Check .env (TAVILY_API_KEY environment variable)
    2. If not found, fetch from Convex settings
    """
    # First, check environment variable
    env_key = os.environ.get("TAVILY_API_KEY")
    if env_key:
        return env_key

    # Fallback to Convex
    convex_key = _get_key_from_convex("tavilyApiKey")
    if convex_key:
        return convex_key

    return None


def _get_key_from_convex(key_name: str) -> Optional[str]:
    """Fetch a specific key from Convex settings."""
    convex_url = os.environ.get("CONVEX_URL")
    if not convex_url:
        return None

    try:
        # Use Convex HTTP API to query settings
        response = httpx.post(
            f"{convex_url}/api/query",
            json={
                "path": "settings:get",
                "args": {},
            },
            timeout=5.0,
        )

        if response.status_code == 200:
            result = response.json()
            # Convex returns { value: { ... settings ... } }
            settings = result.get("value", {})
            return settings.get(key_name)
    except Exception:
        # Silently fail - will use placeholder mode
        pass

    return None


def get_api_keys() -> dict:
    """Get all API keys as a dictionary."""
    return {
        "openrouter": get_openrouter_api_key(),
        "tavily": get_tavily_api_key(),
    }
