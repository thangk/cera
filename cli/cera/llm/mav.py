"""Multi-Agent Verification (MAV) - Cross-model consensus for fact verification."""

from typing import Optional
from dataclasses import dataclass
import asyncio

from cera.prompts import load_and_format


@dataclass
class MAVResult:
    """Result of Multi-Agent Verification."""

    claim: str
    verified: bool
    agreement_score: float
    responses: list[str]
    agreeing_models: list[str]


class MultiAgentVerification:
    """
    Multi-Agent Verification (MAV).

    Uses multiple independent LLMs to verify factual claims
    through 2/3 majority voting with semantic similarity
    threshold τ = 0.85.
    """

    # Default models for MAV (reasoning-capable)
    DEFAULT_MODELS = [
        "claude-sonnet-4",
        "gpt-4o",
        "gemini-2.5-pro",
    ]

    def __init__(
        self,
        api_key: str,
        models: Optional[list[str]] = None,
        similarity_threshold: float = 0.85,
    ):
        self.api_key = api_key
        self.models = models or self.DEFAULT_MODELS
        self.similarity_threshold = similarity_threshold
        self._similarity_model = None  # Lazy-loaded model cache

    def _get_similarity_model(self):
        """Lazy-load and cache the SentenceTransformer model."""
        if self._similarity_model is None:
            import os
            import logging

            # Suppress progress bars and verbose logging
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
            logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

            from sentence_transformers import SentenceTransformer

            self._similarity_model = SentenceTransformer(
                "all-MiniLM-L6-v2", device="cpu"
            )
        return self._similarity_model

    async def _query_model(
        self, model: str, claim: str, context: str
    ) -> str:
        """Query a single model."""
        from .openrouter import OpenRouterClient

        # Load and format the verification prompt
        prompt = load_and_format("mav", "verify", claim=claim, context=context)

        async with OpenRouterClient(self.api_key) as client:
            messages = [
                {
                    "role": "user",
                    "content": prompt,
                },
            ]
            return await client.chat(messages, model=model, temperature=0.0)

    def _compute_similarity(self, text1: str, text2: str) -> float:
        """
        Compute semantic similarity between two texts.

        Uses Sentence-BERT if available, otherwise falls back
        to simple word overlap.
        """
        try:
            from sentence_transformers import util

            model = self._get_similarity_model()  # Use cached model
            embeddings = model.encode(
                [text1, text2], convert_to_tensor=True, show_progress_bar=False
            )
            similarity = util.cos_sim(embeddings[0], embeddings[1])
            return float(similarity[0][0])
        except ImportError:
            # Fallback: Jaccard similarity
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            intersection = words1 & words2
            union = words1 | words2
            return len(intersection) / len(union) if union else 0.0

    def _check_agreement(self, responses: list[str]) -> tuple[bool, float, list[int]]:
        """
        Check if responses agree (2/3 majority).

        Returns:
            Tuple of (agreed, score, agreeing_indices)
        """
        n = len(responses)
        if n < 2:
            return True, 1.0, list(range(n))

        # Compute pairwise similarities
        agreements = []
        for i in range(n):
            for j in range(i + 1, n):
                sim = self._compute_similarity(responses[i], responses[j])
                if sim >= self.similarity_threshold:
                    agreements.append((i, j, sim))

        # Find largest agreeing group
        agreeing = set()
        for i, j, _ in agreements:
            agreeing.add(i)
            agreeing.add(j)

        # 2/3 majority required
        agreement_ratio = len(agreeing) / n
        agreed = agreement_ratio >= 2 / 3

        return agreed, agreement_ratio, list(agreeing)

    async def verify_claim(self, claim: str, context: str = "") -> MAVResult:
        """
        Verify a single claim using MAV.

        Args:
            claim: The claim to verify
            context: Optional context for verification

        Returns:
            MAVResult with verification outcome
        """
        # Query all models in parallel
        tasks = [
            self._query_model(model, claim, context) for model in self.models
        ]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out errors
        valid_responses = []
        valid_models = []
        for model, response in zip(self.models, responses):
            if not isinstance(response, Exception):
                valid_responses.append(response)
                valid_models.append(model)

        # Check agreement
        agreed, score, agreeing_indices = self._check_agreement(valid_responses)
        agreeing_models = [valid_models[i] for i in agreeing_indices]

        return MAVResult(
            claim=claim,
            verified=agreed,
            agreement_score=score,
            responses=valid_responses,
            agreeing_models=agreeing_models,
        )

    async def verify_claims(
        self, claims: list[str], context: str = ""
    ) -> list[MAVResult]:
        """
        Verify multiple claims.

        Args:
            claims: List of claims to verify
            context: Optional shared context

        Returns:
            List of MAVResult for each claim
        """
        tasks = [self.verify_claim(claim, context) for claim in claims]
        return await asyncio.gather(*tasks)

    async def filter_verified(
        self, claims: list[str], context: str = ""
    ) -> list[str]:
        """
        Filter claims to only verified ones.

        Args:
            claims: List of claims to verify
            context: Optional shared context

        Returns:
            List of verified claims only
        """
        results = await self.verify_claims(claims, context)
        return [r.claim for r in results if r.verified]
