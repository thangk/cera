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
                max_tokens=500,
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
                max_tokens=500,
            )

        # Clean up response
        context = response.strip()
        if context.startswith("```"):
            lines = context.split("\n")
            context = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

        logger.info("reviewer_context_extracted", length=len(context))
        return context

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
