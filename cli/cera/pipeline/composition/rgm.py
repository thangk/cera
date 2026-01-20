"""Reviewer Generation Module (RGM) - Synthetic reviewer profile generation."""

import random
from dataclasses import dataclass
from typing import Optional


@dataclass
class ReviewerContext:
    """Context document containing reviewer profile."""

    sex: str
    age: int
    audience_context: str
    style_ref: Optional[str] = None


class ReviewerGenerationModule:
    """
    Reviewer Generation Module (RGM).

    Generates synthetic reviewer profiles through random sampling
    based on configured demographics. No LLM required for this phase.
    """

    def __init__(
        self,
        age_range: tuple[int, int] = (18, 65),
        sex_distribution: dict[str, float] = None,
        audience_contexts: list[str] = None,
    ):
        self.age_range = age_range
        self.sex_distribution = sex_distribution or {
            "male": 0.45,
            "female": 0.45,
            "unspecified": 0.10,
        }
        self.audience_contexts = audience_contexts or ["general user"]

    def generate_profile(self) -> ReviewerContext:
        """
        Generate a single reviewer profile.

        Returns:
            ReviewerContext with randomly sampled demographics
        """
        # Sample sex based on distribution
        rand = random.random()
        cumulative = 0
        sex = "unspecified"
        for s, prob in self.sex_distribution.items():
            cumulative += prob
            if rand < cumulative:
                sex = s
                break

        # Sample age uniformly within range
        age = random.randint(self.age_range[0], self.age_range[1])

        # Sample audience context
        audience = random.choice(self.audience_contexts)

        return ReviewerContext(
            sex=sex,
            age=age,
            audience_context=audience,
        )

    def generate_profiles(self, count: int) -> list[ReviewerContext]:
        """
        Generate multiple reviewer profiles.

        Args:
            count: Number of profiles to generate

        Returns:
            List of ReviewerContext objects
        """
        return [self.generate_profile() for _ in range(count)]
