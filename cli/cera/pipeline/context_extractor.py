"""Context Extractor - Extract subject and reviewer context from reference datasets."""

import json
import random
from typing import Optional

from cera.prompts import load_and_format
from cera.llm.openrouter import OpenRouterClient
from cera.logging import get_logger

logger = get_logger("cera.pipeline.context_extractor")


class ContextExtractor:
    """
    Extract contextual information from reference datasets.

    Used to domain-match generated reviews with real reference data
    for more meaningful MDQA comparisons.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "anthropic/claude-sonnet-4",
        usage_tracker=None,
    ):
        """
        Initialize the context extractor.

        Args:
            api_key: OpenRouter API key
            model: Model to use for extraction (defaults to Claude Sonnet)
            usage_tracker: Optional usage tracker for API costs
        """
        self.api_key = api_key
        self.model = model
        self.usage_tracker = usage_tracker
        logger.info("context_extractor_init", model=model)

    def _sample_reviews(
        self,
        reviews: list[str],
        max_samples: int = 25,
        max_chars: int = 16000,
    ) -> list[str]:
        """
        Sample representative reviews, respecting size limits.

        Args:
            reviews: List of review texts
            max_samples: Maximum number of reviews to sample
            max_chars: Maximum total characters (rough token proxy)

        Returns:
            Sampled list of reviews
        """
        if len(reviews) <= max_samples:
            sampled = reviews.copy()
        else:
            # Random sample for diversity
            sampled = random.sample(reviews, max_samples)

        # Truncate if total chars too high
        total_chars = sum(len(r) for r in sampled)

        if total_chars > max_chars:
            # Sort by length and take shorter ones first
            sampled = sorted(sampled, key=len)
            result = []
            char_count = 0
            for r in sampled:
                if char_count + len(r) > max_chars:
                    break
                result.append(r)
                char_count += len(r)
            return result

        return sampled

    async def extract_subject_context(
        self,
        reviews: list[str],
        sample_count: Optional[int] = None,
    ) -> str:
        """
        Extract subject context from reviews.

        Args:
            reviews: List of review texts to analyze
            sample_count: Optional number of reviews to sample (default: 25)

        Returns:
            Extracted subject context as a paragraph
        """
        sampled = self._sample_reviews(reviews, max_samples=sample_count or 25)
        logger.info("extracting_subject_context", sample_count=len(sampled), total_reviews=len(reviews))

        # Format reviews as numbered list for clarity
        reviews_json = json.dumps(
            [{"review": r} for r in sampled],
            indent=2,
            ensure_ascii=False
        )

        prompt = load_and_format(
            "context_extractor", "subject",
            reviews_json=reviews_json,
        )

        async with OpenRouterClient(self.api_key, usage_tracker=self.usage_tracker) as client:
            response = await client.chat(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=0.3,  # Lower temperature for more consistent extraction
                max_tokens=2048,  # Reasoning models need headroom for thinking tokens
            )

        # Clean up response (remove any markdown formatting if present)
        context = response.strip()
        if context.startswith("```"):
            # Remove code block if LLM wrapped it
            lines = context.split("\n")
            context = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

        logger.info("subject_context_extracted", length=len(context))
        return context

    async def extract_reviewer_context(
        self,
        reviews: list[str],
        sample_count: Optional[int] = None,
    ) -> str:
        """
        Extract reviewer context from reviews.

        Args:
            reviews: List of review texts to analyze
            sample_count: Optional number of reviews to sample (default: 25)

        Returns:
            Extracted reviewer context as a paragraph
        """
        sampled = self._sample_reviews(reviews, max_samples=sample_count or 25)
        logger.info("extracting_reviewer_context", sample_count=len(sampled), total_reviews=len(reviews))

        # Format reviews as numbered list for clarity
        reviews_json = json.dumps(
            [{"review": r} for r in sampled],
            indent=2,
            ensure_ascii=False
        )

        prompt = load_and_format(
            "context_extractor", "reviewer",
            reviews_json=reviews_json,
        )

        async with OpenRouterClient(self.api_key, usage_tracker=self.usage_tracker) as client:
            response = await client.chat(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=0.3,
                max_tokens=2048,  # Reasoning models need headroom for thinking tokens
            )

        # Clean up response
        context = response.strip()
        if context.startswith("```"):
            lines = context.split("\n")
            context = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

        logger.info("reviewer_context_extracted", length=len(context))
        return context

    async def extract_subject_query(
        self,
        reviews: list[str],
        sample_count: Optional[int] = None,
    ) -> str:
        """
        Extract a concise subject query from reviews.

        Args:
            reviews: List of review texts to analyze
            sample_count: Optional number of reviews to sample (default: 25)

        Returns:
            Concise subject query (3-8 words)
        """
        sampled = self._sample_reviews(reviews, max_samples=sample_count or 25)
        logger.info("extracting_subject_query", sample_count=len(sampled), total_reviews=len(reviews))

        # Format reviews as JSON for the prompt
        reviews_json = json.dumps(
            [{"review": r} for r in sampled],
            indent=2,
            ensure_ascii=False
        )

        prompt = load_and_format(
            "context_extractor", "subject_query",
            reviews_json=reviews_json,
        )

        async with OpenRouterClient(self.api_key, usage_tracker=self.usage_tracker) as client:
            response = await client.chat(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=0.3,
                max_tokens=1024,  # Reasoning models need headroom for thinking tokens
            )

        # Clean up response - just the query, no quotes or extra text
        query = response.strip().strip('"\'')

        # Fallback if response is too long or empty
        if not query or len(query) > 100:
            query = "General product or service reviews"

        logger.info("subject_query_extracted", query=query)
        return query

    async def extract_domain(
        self,
        reviews: list[str],
        sample_count: Optional[int] = None,
    ) -> dict:
        """
        Extract the domain/category from reviews.

        Args:
            reviews: List of review texts to analyze
            sample_count: Optional number of reviews to sample (default: 25)

        Returns:
            Dict with 'value' (domain string) and 'confidence' (0.0-1.0)
        """
        sampled = self._sample_reviews(reviews, max_samples=sample_count or 25)
        logger.info("extracting_domain", sample_count=len(sampled), total_reviews=len(reviews))

        reviews_json = json.dumps(
            [{"review": r} for r in sampled],
            indent=2,
            ensure_ascii=False
        )

        prompt = load_and_format(
            "context_extractor", "domain",
            reviews_json=reviews_json,
        )

        async with OpenRouterClient(self.api_key, usage_tracker=self.usage_tracker) as client:
            response = await client.chat(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=0.3,
                max_tokens=1024,  # Reasoning models need headroom for thinking tokens
            )

        # Parse JSON response using robust extractor
        try:
            from cera.api import _extract_json_from_llm
            result = _extract_json_from_llm(response, expected_type="object")
            domain = result.get("domain", "General")
            confidence = float(result.get("confidence", 0.5))
        except Exception:
            logger.warning("domain_extraction_parse_error", response=response[:200])
            domain = "General"
            confidence = 0.3

        logger.info("domain_extracted", domain=domain, confidence=confidence)
        return {"value": domain, "confidence": confidence}

    async def extract_region(
        self,
        reviews: list[str],
        sample_count: Optional[int] = None,
    ) -> dict:
        """
        Extract the geographic region from reviews.

        Args:
            reviews: List of review texts to analyze
            sample_count: Optional number of reviews to sample (default: 25)

        Returns:
            Dict with 'value' (region string or None), 'confidence', and optional 'reason'
        """
        sampled = self._sample_reviews(reviews, max_samples=sample_count or 25)
        logger.info("extracting_region", sample_count=len(sampled), total_reviews=len(reviews))

        reviews_json = json.dumps(
            [{"review": r} for r in sampled],
            indent=2,
            ensure_ascii=False
        )

        prompt = load_and_format(
            "context_extractor", "region",
            reviews_json=reviews_json,
        )

        async with OpenRouterClient(self.api_key, usage_tracker=self.usage_tracker) as client:
            response = await client.chat(
                messages=[{"role": "user", "content": prompt}],
                model=self.model,
                temperature=0.3,
                max_tokens=1024,  # Reasoning models need headroom for thinking tokens
            )

        # Parse JSON response using robust extractor
        try:
            from cera.api import _extract_json_from_llm
            result = _extract_json_from_llm(response, expected_type="object")
            region = result.get("region")  # Can be None
            confidence = float(result.get("confidence", 0.0))
            reason = result.get("reason")
        except Exception:
            logger.warning("region_extraction_parse_error", response=response[:200])
            region = None
            confidence = 0.0
            reason = "Failed to parse LLM response"

        logger.info("region_extracted", region=region, confidence=confidence)
        return {"value": region, "confidence": confidence, "reason": reason}

    async def extract_both(
        self,
        reviews: list[str],
        sample_count: Optional[int] = None,
    ) -> tuple[str, str]:
        """
        Extract both subject and reviewer context.

        Args:
            reviews: List of review texts to analyze
            sample_count: Optional number of reviews to sample (default: 25)

        Returns:
            Tuple of (subject_context, reviewer_context)
        """
        subject_context = await self.extract_subject_context(reviews, sample_count)
        reviewer_context = await self.extract_reviewer_context(reviews, sample_count)
        return subject_context, reviewer_context
