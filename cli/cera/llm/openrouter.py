"""OpenRouter Client - Provider-agnostic LLM access."""

from typing import Optional, AsyncIterator
import httpx


# Models that support the :online suffix for web search
# Each provider uses their own search infrastructure
ONLINE_CAPABLE_MODELS = {
    # Anthropic - uses Anthropic Web Search
    "anthropic/claude-sonnet-4",
    "anthropic/claude-opus-4",
    "anthropic/claude-3.5-sonnet",
    "anthropic/claude-3.5-haiku",
    # OpenAI - uses Exa search
    "openai/gpt-4o",
    "openai/gpt-4o-mini",
    "openai/gpt-4-turbo",
    "openai/o1",
    "openai/o1-mini",
    "openai/o3-mini",
    # Google - uses Google Search (grounding)
    "google/gemini-2.5-pro-preview",
    "google/gemini-2.5-flash-preview",
    "google/gemini-2.0-flash-001",
    "google/gemini-pro",
    # xAI - uses Grok's search
    "x-ai/grok-3",
    "x-ai/grok-3-mini",
    # Mistral
    "mistralai/mistral-large",
    "mistralai/mistral-medium",
}

# Models with NATIVE search (always have search, no :online needed)
NATIVE_SEARCH_MODELS = {
    "perplexity/sonar",
    "perplexity/sonar-pro",
    "perplexity/sonar-reasoning",
    "perplexity/sonar-deep-research",
}


def supports_online_search(model: str) -> bool:
    """
    Check if a model supports web search.

    Args:
        model: Model ID (e.g., "anthropic/claude-sonnet-4")

    Returns:
        True if model supports :online suffix or has native search
    """
    # Strip any existing :online suffix
    base_model = model.replace(":online", "")

    # Check native search first
    if base_model in NATIVE_SEARCH_MODELS:
        return True

    # Check online-capable models
    return base_model in ONLINE_CAPABLE_MODELS


def get_search_model_id(model: str) -> str:
    """
    Get the model ID configured for web search.

    Args:
        model: Base model ID

    Returns:
        Model ID with :online suffix if needed, or native search model as-is
    """
    base_model = model.replace(":online", "")

    if base_model in NATIVE_SEARCH_MODELS:
        # Native search models don't need suffix
        return base_model

    if base_model in ONLINE_CAPABLE_MODELS:
        # Add :online suffix for web search
        return f"{base_model}:online"

    # Model doesn't support web search
    return None


class OpenRouterClient:
    """
    OpenRouter API Client.

    Provides unified access to multiple LLM providers:
    - Anthropic (Claude models)
    - OpenAI (GPT models)
    - Google (Gemini models)
    - xAI (Grok models)
    - DeepSeek
    - And more...
    """

    BASE_URL = "https://openrouter.ai/api/v1"

    # Common model mappings
    MODELS = {
        # Anthropic
        "claude-opus-4": "anthropic/claude-opus-4",
        "claude-sonnet-4": "anthropic/claude-sonnet-4",
        "claude-haiku-3.5": "anthropic/claude-3.5-haiku",
        # OpenAI
        "gpt-4o": "openai/gpt-4o",
        "gpt-4o-mini": "openai/gpt-4o-mini",
        "gpt-4-turbo": "openai/gpt-4-turbo",
        # Google
        "gemini-2.5-pro": "google/gemini-2.5-pro-preview",
        "gemini-2.5-flash": "google/gemini-2.5-flash-preview",
        # xAI
        "grok-3": "x-ai/grok-3",
        # DeepSeek
        "deepseek-chat": "deepseek/deepseek-chat",
        "deepseek-r1": "deepseek/deepseek-r1",
    }

    def __init__(self, api_key: str, site_url: str = "", site_name: str = "CERA"):
        self.api_key = api_key
        self.site_url = site_url
        self.site_name = site_name
        self.client = httpx.AsyncClient()

    def _get_model_id(self, model: str) -> str:
        """Get full model ID from short name or return as-is."""
        return self.MODELS.get(model, model)

    async def chat(
        self,
        messages: list[dict],
        model: str = "claude-sonnet-4",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        stream: bool = False,
    ) -> str | AsyncIterator[str]:
        """
        Send a chat completion request.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model name (short or full)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response

        Returns:
            Generated text or async iterator if streaming
        """
        model_id = self._get_model_id(model)

        response = await self.client.post(
            f"{self.BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": self.site_url,
                "X-Title": self.site_name,
                "Content-Type": "application/json",
            },
            json={
                "model": model_id,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": stream,
            },
            timeout=120.0,
        )
        response.raise_for_status()

        data = response.json()
        return data["choices"][0]["message"]["content"]

    async def generate_review(
        self,
        subject: str,
        features: list[str],
        polarity: str,
        reviewer_age: int,
        reviewer_context: str,
        temperature: float = 0.8,
        model: str = "claude-sonnet-4",
    ) -> str:
        """
        Generate a single review.

        Args:
            subject: Product/service being reviewed
            features: Key features to mention
            polarity: Target sentiment (positive/neutral/negative)
            reviewer_age: Reviewer's age
            reviewer_context: Reviewer's context/persona
            temperature: Sampling temperature
            model: Model to use

        Returns:
            Generated review text
        """
        system_prompt = """You are a review writer creating authentic-sounding product reviews.
Your reviews should:
- Sound natural and human-written
- Include specific details about the product
- Match the requested sentiment accurately
- Vary in style and structure
- Be 2-5 sentences long"""

        user_prompt = f"""Write a {polarity} review for {subject}.

Key features: {', '.join(features)}
Reviewer: {reviewer_age}-year-old {reviewer_context}

Write only the review text, nothing else."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        return await self.chat(messages, model=model, temperature=temperature)

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
