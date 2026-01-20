"""Web Search Client - Search the web for factual information."""

from dataclasses import dataclass
from typing import Optional
import httpx


@dataclass
class SearchResult:
    """A single search result."""

    title: str
    url: str
    content: str
    score: float = 0.0


@dataclass
class SearchResponse:
    """Web search response."""

    query: str
    results: list[SearchResult]
    answer: Optional[str] = None  # Some APIs provide a direct answer


class TavilySearchClient:
    """
    Tavily Search API client.

    Tavily is designed specifically for AI use cases and provides
    clean, relevant search results optimized for LLM consumption.
    """

    BASE_URL = "https://api.tavily.com"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = httpx.AsyncClient()

    async def search(
        self,
        query: str,
        search_depth: str = "advanced",
        max_results: int = 10,
        include_answer: bool = True,
        include_raw_content: bool = False,
    ) -> SearchResponse:
        """
        Search the web using Tavily API.

        Args:
            query: Search query
            search_depth: "basic" or "advanced" (more thorough)
            max_results: Maximum number of results
            include_answer: Include AI-generated answer
            include_raw_content: Include full page content

        Returns:
            SearchResponse with results
        """
        response = await self.client.post(
            f"{self.BASE_URL}/search",
            json={
                "api_key": self.api_key,
                "query": query,
                "search_depth": search_depth,
                "max_results": max_results,
                "include_answer": include_answer,
                "include_raw_content": include_raw_content,
            },
            timeout=30.0,
        )
        response.raise_for_status()
        data = response.json()

        results = [
            SearchResult(
                title=r.get("title", ""),
                url=r.get("url", ""),
                content=r.get("content", ""),
                score=r.get("score", 0.0),
            )
            for r in data.get("results", [])
        ]

        return SearchResponse(
            query=query,
            results=results,
            answer=data.get("answer"),
        )

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


class PerplexitySearchClient:
    """
    Perplexity API client via OpenRouter.

    Uses Perplexity's models which have built-in web search capability.
    This is a fallback when Tavily API key is not available.
    """

    OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

    # Perplexity models with online search capability
    SEARCH_MODELS = [
        "perplexity/sonar",  # Good balance
        "perplexity/sonar-pro",  # Most capable
        "perplexity/sonar-reasoning",  # For complex queries
    ]

    def __init__(self, api_key: str, model: str = "perplexity/sonar"):
        """
        Initialize Perplexity search client.

        Args:
            api_key: OpenRouter API key
            model: Perplexity model to use
        """
        self.api_key = api_key
        self.model = model
        self.client = httpx.AsyncClient()

    async def search(
        self,
        query: str,
        max_results: int = 10,
    ) -> SearchResponse:
        """
        Search using Perplexity's built-in web search.

        Args:
            query: Search query
            max_results: Hint for number of sources (not strictly enforced)

        Returns:
            SearchResponse with synthesized answer and sources
        """
        system_prompt = f"""You are a research assistant with web search capabilities.
Search the web and provide factual, well-sourced information.
Include {max_results} relevant sources when possible.
Format your response with clear facts that can be verified."""

        response = await self.client.post(
            self.OPENROUTER_URL,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "X-Title": "CERA",
            },
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query},
                ],
                "temperature": 0.0,
            },
            timeout=60.0,
        )
        response.raise_for_status()
        data = response.json()

        answer = data["choices"][0]["message"]["content"]

        # Perplexity doesn't return structured results like Tavily
        # We return the answer as a single "result" with synthesized content
        return SearchResponse(
            query=query,
            results=[
                SearchResult(
                    title="Perplexity Search Result",
                    url="",
                    content=answer,
                    score=1.0,
                )
            ],
            answer=answer,
        )

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


class WebSearchClient:
    """
    Unified web search client.

    Automatically selects the best available search backend:
    1. Tavily (if API key provided) - preferred for structured results
    2. Perplexity via OpenRouter - fallback with built-in search
    """

    def __init__(
        self,
        openrouter_api_key: str,
        tavily_api_key: Optional[str] = None,
        perplexity_model: str = "perplexity/sonar",
    ):
        """
        Initialize web search client.

        Args:
            openrouter_api_key: OpenRouter API key (required)
            tavily_api_key: Optional Tavily API key (preferred)
            perplexity_model: Perplexity model to use as fallback
        """
        self.openrouter_api_key = openrouter_api_key
        self.tavily_api_key = tavily_api_key
        self.perplexity_model = perplexity_model

        # Select backend
        if tavily_api_key:
            self._client = TavilySearchClient(tavily_api_key)
            self.backend = "tavily"
        else:
            self._client = PerplexitySearchClient(openrouter_api_key, perplexity_model)
            self.backend = "perplexity"

    async def search(
        self,
        query: str,
        max_results: int = 10,
    ) -> SearchResponse:
        """
        Search the web.

        Args:
            query: Search query
            max_results: Maximum number of results

        Returns:
            SearchResponse with results
        """
        return await self._client.search(query, max_results=max_results)

    async def search_product(
        self,
        product_name: str,
        region: str = "united states",
        category: str = "general",
    ) -> SearchResponse:
        """
        Search for product information.

        Constructs an optimized search query for product research.

        Args:
            product_name: Name of the product
            region: Geographic region for context
            category: Product category

        Returns:
            SearchResponse with product information
        """
        query = f"{product_name} {category} features pros cons reviews {region}"
        return await self.search(query, max_results=10)

    async def close(self):
        """Close the client."""
        await self._client.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
