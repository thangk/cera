"""OpenRouter Client - Provider-agnostic LLM access."""

from dataclasses import dataclass
from typing import Optional, AsyncIterator
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception

from cera.logging import get_logger

logger = get_logger("cera.llm.openrouter")


def _is_retryable(exc: BaseException) -> bool:
    """Check if an exception is retryable (429, 500, 502, 503)."""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in (429, 500, 502, 503)
    return False


@dataclass
class ChatResponse:
    """Response from an OpenRouter chat completion."""

    content: str
    usage: Optional[dict] = None  # {prompt_tokens, completion_tokens, total_tokens}
    model: Optional[str] = None  # actual model used
    id: Optional[str] = None  # generation ID


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

    All models support web search via OpenRouter's :online suffix,
    which uses native provider search (Anthropic, Google, OpenAI, xAI)
    or Exa as a fallback for other models.
    """
    return True


def get_search_model_id(model: str) -> str:
    """
    Get the model ID configured for web search.

    Args:
        model: Base model ID

    Returns:
        Model ID with :online suffix, or native search model as-is
    """
    base_model = model.replace(":online", "")

    if base_model in NATIVE_SEARCH_MODELS:
        # Native search models don't need suffix
        return base_model

    # All models support :online via OpenRouter (native or Exa fallback)
    return f"{base_model}:online"


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

    def __init__(self, api_key: str, site_url: str = "", site_name: str = "CERA", usage_tracker=None):
        self.api_key = api_key
        self.site_url = site_url
        self.site_name = site_name
        self.usage_tracker = usage_tracker
        self.client = httpx.AsyncClient()

    def _get_model_id(self, model: str) -> str:
        """Get full model ID from short name or return as-is."""
        return self.MODELS.get(model, model)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception(_is_retryable),
        before_sleep=lambda retry_state: logger.warning(
            "llm_retry",
            attempt=retry_state.attempt_number,
            wait=retry_state.next_action.sleep,
            error=str(retry_state.outcome.exception()),
        ),
    )
    async def chat(
        self,
        messages: list[dict],
        model: str = "claude-sonnet-4",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        stream: bool = False,
        reasoning: dict | None = None,
    ) -> str | AsyncIterator[str]:
        """
        Send a chat completion request.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model name (short or full)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            reasoning: Optional reasoning config, e.g. {"effort": "high"}

        Returns:
            Generated text or async iterator if streaming
        """
        model_id = self._get_model_id(model)

        body = {
            "model": model_id,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        if reasoning:
            body["reasoning"] = reasoning

        response = await self.client.post(
            f"{self.BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": self.site_url,
                "X-Title": self.site_name,
                "Content-Type": "application/json",
            },
            json=body,
            timeout=180.0,
        )
        response.raise_for_status()

        data = response.json()

        # Handle API error responses that don't have 'choices'
        if "choices" not in data:
            error_msg = data.get("error", {}).get("message") or data.get("error") or str(data)
            logger.error("openrouter_api_error", model=model_id, error=error_msg)
            raise ValueError(f"OpenRouter API error: {error_msg}")

        message = data["choices"][0]["message"]
        content = message.get("content") or ""

        # Reasoning models (e.g., Perplexity Sonar Reasoning Pro, DeepSeek R1)
        # may return empty content with output in reasoning fields instead
        if not content:
            reasoning_content = message.get("reasoning_content") or message.get("reasoning") or ""
            if reasoning_content:
                logger.info("reasoning_model_content_fallback", model=model_id)
                content = reasoning_content

        # Record usage if tracker is available
        if self.usage_tracker:
            usage = data.get("usage")
            if usage:
                from cera.llm.usage import LLMUsage
                self.usage_tracker.record(LLMUsage(
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    total_tokens=usage.get("total_tokens", 0),
                    model=model_id,
                ))

        logger.debug("llm_call", model=model_id, tokens=data.get("usage", {}).get("total_tokens"))

        return content

    async def chat_with_usage(
        self,
        messages: list[dict],
        model: str = "claude-sonnet-4",
        temperature: float = 0.7,
        max_tokens: int = 4096,
    ) -> ChatResponse:
        """
        Send a chat completion request and return full response with usage data.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model name (short or full)
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            ChatResponse with content, usage, model, and generation ID
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
            },
            timeout=120.0,
        )
        response.raise_for_status()

        data = response.json()

        # Handle API error responses that don't have 'choices'
        if "choices" not in data:
            error_msg = data.get("error", {}).get("message") or data.get("error") or str(data)
            logger.error("openrouter_api_error", model=model_id, error=error_msg)
            raise ValueError(f"OpenRouter API error: {error_msg}")

        message = data["choices"][0]["message"]
        content = message.get("content") or ""

        # Reasoning model fallback (same as chat())
        if not content:
            reasoning_content = message.get("reasoning_content") or message.get("reasoning") or ""
            if reasoning_content:
                logger.info("reasoning_model_content_fallback", model=model_id)
                content = reasoning_content

        usage = data.get("usage")
        gen_id = data.get("id")
        actual_model = data.get("model", model_id)

        # Record usage if tracker is available
        if self.usage_tracker and usage:
            from cera.llm.usage import LLMUsage
            self.usage_tracker.record(LLMUsage(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
                model=actual_model,
            ))

        return ChatResponse(
            content=content,
            usage=usage,
            model=actual_model,
            id=gen_id,
        )

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
