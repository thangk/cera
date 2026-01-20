"""Attributes Composition Module (ACM) - Review attributes configuration."""

from dataclasses import dataclass


@dataclass
class NoiseConfig:
    """Configuration for noise injection."""

    typo_rate: float = 0.01
    colloquialism: bool = True
    grammar_errors: bool = True


@dataclass
class PolarityDistribution:
    """Distribution of sentiment polarities."""

    positive: float = 0.65
    neutral: float = 0.15
    negative: float = 0.20

    def validate(self) -> bool:
        """Validate that distribution sums to 1.0."""
        total = self.positive + self.neutral + self.negative
        return abs(total - 1.0) < 0.01


@dataclass
class AttributesContext:
    """Context document containing review attributes."""

    polarity_distribution: PolarityDistribution
    noise_config: NoiseConfig
    length_sentences: tuple[int, int]
    temperature: tuple[float, float]


class AttributesCompositionModule:
    """
    Attributes Composition Module (ACM).

    Configures review generation parameters including polarity
    distribution, noise settings, and generation constraints.
    No LLM required - direct configuration parsing.
    """

    def __init__(
        self,
        polarity: dict[str, float] = None,
        noise: dict = None,
        length_range: tuple[int, int] = (2, 5),
        temp_range: tuple[float, float] = (0.7, 0.9),
    ):
        # Parse polarity
        if polarity:
            self.polarity = PolarityDistribution(
                positive=polarity.get("positive", 0.65),
                neutral=polarity.get("neutral", 0.15),
                negative=polarity.get("negative", 0.20),
            )
        else:
            self.polarity = PolarityDistribution()

        # Parse noise config
        if noise:
            self.noise = NoiseConfig(
                typo_rate=noise.get("typo_rate", 0.01),
                colloquialism=noise.get("colloquialism", True),
                grammar_errors=noise.get("grammar_errors", True),
            )
        else:
            self.noise = NoiseConfig()

        self.length_range = length_range
        self.temp_range = temp_range

    def compose(self) -> AttributesContext:
        """
        Compose the attributes context.

        Returns:
            AttributesContext with all configuration parameters
        """
        return AttributesContext(
            polarity_distribution=self.polarity,
            noise_config=self.noise,
            length_sentences=self.length_range,
            temperature=self.temp_range,
        )

    @classmethod
    def realistic_preset(cls) -> "AttributesCompositionModule":
        """
        Create ACM with realistic polarity distribution.

        Based on empirical Amazon review distribution:
        65% positive, 15% neutral, 20% negative.
        """
        return cls(
            polarity={"positive": 0.65, "neutral": 0.15, "negative": 0.20},
        )

    @classmethod
    def balanced_preset(cls) -> "AttributesCompositionModule":
        """
        Create ACM with balanced polarity distribution.

        Equal distribution for class-imbalance experiments:
        33% positive, 34% neutral, 33% negative.
        """
        return cls(
            polarity={"positive": 0.33, "neutral": 0.34, "negative": 0.33},
        )
