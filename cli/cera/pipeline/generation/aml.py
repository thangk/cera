"""Authenticity Modeling Layer (AML) - Review generation with authenticity control."""

from dataclasses import dataclass
from typing import Optional
import random
import asyncio


@dataclass
class GeneratedReview:
    """A generated synthetic review."""

    text: str
    polarity: str  # "positive", "neutral", "negative"
    aspects: list[dict]
    reviewer_age: int
    reviewer_sex: str
    audience_context: str
    model: str
    temperature: float
    batch_id: int


class AuthenticityModelingLayer:
    """
    Authenticity Modeling Layer (AML).

    Combines subject context, reviewer profiles, and attributes
    to generate authentic-sounding synthetic reviews with
    controlled polarity distribution.
    """

    def __init__(self, api_key: str, provider: str = "anthropic", model: str = "claude-sonnet-4"):
        self.api_key = api_key
        self.provider = provider
        self.model = model
        self._client = None

    def _is_placeholder_mode(self) -> bool:
        """Check if we're in placeholder mode (no real API key)."""
        return not self.api_key or self.api_key == "placeholder"

    async def _get_client(self):
        """Get or create the OpenRouter client."""
        if self._client is None:
            from cera.llm.openrouter import OpenRouterClient
            self._client = OpenRouterClient(self.api_key)
        return self._client

    def _determine_polarity(self, polarity_dist: dict) -> str:
        """Determine review polarity based on distribution."""
        rand = random.random()
        neg_threshold = polarity_dist.get("negative", 0.20)
        neu_threshold = neg_threshold + polarity_dist.get("neutral", 0.15)

        if rand < neg_threshold:
            return "negative"
        elif rand < neu_threshold:
            return "neutral"
        else:
            return "positive"

    def _build_system_prompt(self, subject_context: dict) -> str:
        """Build the system prompt for review generation."""
        subject = subject_context.get("subject", "Product")
        features = subject_context.get("features", [])
        pros = subject_context.get("pros", [])
        cons = subject_context.get("cons", [])

        prompt = f"""You are an authentic review writer creating realistic product/service reviews for {subject}.

Your reviews should:
- Sound completely natural and human-written
- Include specific, believable details about the experience
- Match the requested sentiment accurately and naturally
- Vary in style, structure, and length (2-5 sentences)
- Avoid generic phrases like "I highly recommend" - be specific
- Never mention that you're an AI or that this is a generated review

Key features of {subject}: {', '.join(features[:5]) if features else 'various features'}
Known pros: {', '.join(pros[:3]) if pros else 'quality, value'}
Known cons: {', '.join(cons[:3]) if cons else 'minor issues'}"""

        return prompt

    def _build_user_prompt(
        self,
        subject_context: dict,
        reviewer_context: dict,
        polarity: str,
    ) -> str:
        """Build the user prompt for review generation."""
        subject = subject_context.get("subject", "Product")
        age = reviewer_context.get("age", 30)
        sex = reviewer_context.get("sex", "unspecified")
        context = reviewer_context.get("audience_context", "general user")

        sentiment_guide = {
            "positive": "enthusiastic and satisfied, highlighting what you loved",
            "neutral": "balanced and objective, mentioning both good and bad aspects",
            "negative": "disappointed and critical, explaining what went wrong",
        }

        prompt = f"""Write a {polarity} review for {subject}.

Reviewer persona: {age}-year-old {sex if sex != 'unspecified' else 'person'}, {context}
Tone: {sentiment_guide.get(polarity, 'balanced')}

Write ONLY the review text. No labels, no quotation marks, no explanations."""

        return prompt

    async def generate_review(
        self,
        subject_context: dict,
        reviewer_context: dict,
        attributes_context: dict,
        batch_id: int = 0,
    ) -> GeneratedReview:
        """
        Generate a single synthetic review.

        Args:
            subject_context: Subject intelligence data
            reviewer_context: Reviewer profile data
            attributes_context: Generation attributes
            batch_id: Batch identifier for tracking

        Returns:
            GeneratedReview with generated content
        """
        # Determine polarity based on distribution
        polarity_dist = attributes_context.get("polarity_distribution", {})
        polarity = self._determine_polarity(polarity_dist)

        # Sample temperature
        temp_range = attributes_context.get("temperature", (0.7, 0.9))
        temperature = random.uniform(temp_range[0], temp_range[1])

        # Generate review text
        if self._is_placeholder_mode():
            # Placeholder mode for testing
            review_text = self._generate_placeholder_review(
                subject_context.get("subject", "Product"),
                polarity,
                reviewer_context.get("audience_context", "user"),
            )
        else:
            # Real LLM generation via OpenRouter
            review_text = await self._generate_with_llm(
                subject_context,
                reviewer_context,
                polarity,
                temperature,
            )

        return GeneratedReview(
            text=review_text,
            polarity=polarity,
            aspects=[],  # TODO: Extract aspects from generated text
            reviewer_age=reviewer_context.get("age", 30),
            reviewer_sex=reviewer_context.get("sex", "unspecified"),
            audience_context=reviewer_context.get("audience_context", "general"),
            model=self.model,
            temperature=temperature,
            batch_id=batch_id,
        )

    async def _generate_with_llm(
        self,
        subject_context: dict,
        reviewer_context: dict,
        polarity: str,
        temperature: float,
    ) -> str:
        """Generate review using OpenRouter LLM."""
        client = await self._get_client()

        system_prompt = self._build_system_prompt(subject_context)
        user_prompt = self._build_user_prompt(subject_context, reviewer_context, polarity)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = await client.chat(
                messages=messages,
                model=self.model,
                temperature=temperature,
                max_tokens=500,
            )
            # Clean up response (remove quotes if present)
            text = response.strip().strip('"').strip("'")
            return text
        except Exception as e:
            # Fallback to placeholder on error
            print(f"Warning: LLM generation failed: {e}")
            return self._generate_placeholder_review(
                subject_context.get("subject", "Product"),
                polarity,
                reviewer_context.get("audience_context", "user"),
            )

    def _generate_placeholder_review(
        self, subject: str, polarity: str, audience: str
    ) -> str:
        """Generate a placeholder review for testing."""
        templates = {
            "positive": [
                f"I absolutely love the {subject}! It exceeded all my expectations.",
                f"As a {audience}, I can confidently say the {subject} is fantastic.",
                f"The {subject} has been a great purchase. Highly recommended!",
                f"Really impressed with the {subject}. Quality is excellent.",
                f"Best decision I made was getting the {subject}. Worth every penny.",
            ],
            "neutral": [
                f"The {subject} is decent. It does what it's supposed to do.",
                f"My experience with the {subject} has been mixed.",
                f"The {subject} has both pros and cons worth considering.",
                f"It's an okay {subject}. Nothing special but gets the job done.",
                f"Average {subject}. Some good features but room for improvement.",
            ],
            "negative": [
                f"I'm disappointed with the {subject}. It didn't meet my expectations.",
                f"The {subject} has several issues that need addressing.",
                f"I regret purchasing the {subject}. Not worth the price.",
                f"Would not recommend the {subject}. Too many problems.",
                f"Frustrated with the {subject}. Expected much better quality.",
            ],
        }
        return random.choice(templates.get(polarity, templates["neutral"]))

    async def generate_batch(
        self,
        subject_context: dict,
        reviewer_contexts: list[dict],
        attributes_context: dict,
        batch_id: int = 0,
        concurrent_requests: int = 5,
    ) -> list[GeneratedReview]:
        """
        Generate a batch of synthetic reviews.

        Args:
            subject_context: Subject intelligence data
            reviewer_contexts: List of reviewer profiles
            attributes_context: Generation attributes
            batch_id: Batch identifier
            concurrent_requests: Max concurrent LLM requests

        Returns:
            List of GeneratedReview objects
        """
        if self._is_placeholder_mode():
            # Sequential generation for placeholder mode (fast)
            reviews = []
            for reviewer in reviewer_contexts:
                review = await self.generate_review(
                    subject_context,
                    reviewer,
                    attributes_context,
                    batch_id,
                )
                reviews.append(review)
            return reviews

        # Concurrent generation for real LLM calls
        semaphore = asyncio.Semaphore(concurrent_requests)

        async def generate_with_semaphore(reviewer):
            async with semaphore:
                return await self.generate_review(
                    subject_context,
                    reviewer,
                    attributes_context,
                    batch_id,
                )

        tasks = [generate_with_semaphore(reviewer) for reviewer in reviewer_contexts]
        reviews = await asyncio.gather(*tasks)
        return list(reviews)

    async def close(self):
        """Close the OpenRouter client if open."""
        if self._client:
            await self._client.close()
            self._client = None
