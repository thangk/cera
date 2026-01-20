"""LLM Module - Language model abstractions and verification."""

from .openrouter import (
    OpenRouterClient,
    supports_online_search,
    get_search_model_id,
    ONLINE_CAPABLE_MODELS,
    NATIVE_SEARCH_MODELS,
)
from .mav import MultiAgentVerification
from .web_search import WebSearchClient, TavilySearchClient, PerplexitySearchClient

__all__ = [
    "OpenRouterClient",
    "supports_online_search",
    "get_search_model_id",
    "ONLINE_CAPABLE_MODELS",
    "NATIVE_SEARCH_MODELS",
    "MultiAgentVerification",
    "WebSearchClient",
    "TavilySearchClient",
    "PerplexitySearchClient",
]
