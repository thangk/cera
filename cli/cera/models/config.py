"""Configuration Models - Pydantic schemas for CERA configuration."""

from pydantic import BaseModel, Field
from typing import Optional


class MAVConfig(BaseModel):
    """Multi-Agent Verification configuration for SIL."""

    enabled: bool = Field(default=True, description="Enable MAV verification (requires 2+ models)")
    models: list[str] = Field(
        default_factory=list,
        min_length=2,
        description="Models for research. 2+ models = MAV (ceil(2/3*N) majority voting)",
    )
    similarity_threshold: float = Field(
        default=0.85, ge=0, le=1, description="Query deduplication threshold"
    )
    answer_threshold: float = Field(
        default=0.80, ge=0, le=1, description="Per-query answer consensus threshold"
    )
    max_queries: int = Field(
        default=30, ge=5, le=100, description="Soft cap on pooled queries after deduplication"
    )


class SubjectProfile(BaseModel):
    """Subject configuration for review generation."""

    query: str = Field(..., description="Subject to research (e.g., 'iPhone 15 Pro')")
    additional_context: Optional[str] = Field(default=None, description="Additional context about the subject (e.g., 'Limited edition with only 1000 units produced')")
    region: str = Field(default="united states", description="Geographic region")
    domain: str = Field(default="general", description="Product/service domain (e.g., restaurant, laptop, hotel)")
    aspect_categories: Optional[list[str]] = Field(default=None, description="Aspect categories for ABSA annotation (e.g., FOOD#QUALITY, SERVICE#GENERAL)")
    feature_count: str = Field(default="5-10", description="Range of features to extract")
    sentiment_depth: str = Field(
        default="praise and complain", description="Level of sentiment analysis"
    )
    mav: Optional[MAVConfig] = Field(default=None, description="MAV configuration for SIL (required for web, optional for CLI)")
    custom_schema: Optional[dict] = Field(
        default=None, description="Custom extraction schema"
    )
    seed_subject: Optional[str] = Field(
        default=None, description="Path to seed subject JSON"
    )


class SexDistribution(BaseModel):
    """Sex distribution for reviewer profiles."""

    male: float = Field(default=0.45, ge=0, le=1)
    female: float = Field(default=0.45, ge=0, le=1)
    unspecified: float = Field(default=0.10, ge=0, le=1)


class ReviewerProfile(BaseModel):
    """Reviewer profile configuration."""

    age_range: tuple[int, int] = Field(default=(18, 65), description="Age range")
    sex_distribution: SexDistribution = Field(default_factory=SexDistribution)
    additional_context: Optional[str] = Field(
        default=None,
        description="Additional context about typical reviewers (e.g., demographics, expectations, behaviors). Passed verbatim to generation prompt.",
    )
    seed_datasets: Optional[list[str]] = Field(
        default=None, description="Paths to seed datasets for style matching"
    )


class PolarityDistribution(BaseModel):
    """Polarity distribution for generated reviews."""

    positive: float = Field(default=0.65, ge=0, le=1)
    neutral: float = Field(default=0.15, ge=0, le=1)
    negative: float = Field(default=0.20, ge=0, le=1)


class NoiseConfig(BaseModel):
    """Noise injection configuration."""

    typo_rate: float = Field(default=0.01, ge=0, le=0.1, description="Typo rate (0-10%)")
    colloquialism: bool = Field(default=True, description="Enable colloquialisms")
    grammar_errors: bool = Field(default=True, description="Enable minor grammar errors")
    preset: Optional[str] = Field(
        default=None,
        description="Noise preset: 'none', 'light', 'moderate', 'heavy'. If set, overrides other noise settings."
    )
    advanced: bool = Field(default=False, description="Enable advanced noise (OCR, contextual)")
    use_ocr: bool = Field(default=False, description="Enable OCR-style character errors")
    use_contextual: bool = Field(default=False, description="Enable BERT-based contextual substitution")


class AttributesProfile(BaseModel):
    """Review attributes configuration."""

    polarity: PolarityDistribution = Field(default_factory=PolarityDistribution)
    noise: NoiseConfig = Field(default_factory=NoiseConfig)
    length_range: tuple[int, int] = Field(
        default=(2, 5), description="Sentence count range"
    )
    temp_range: tuple[float, float] = Field(
        default=(0.7, 0.9), description="Temperature range"
    )


class GenerationConfig(BaseModel):
    """Generation settings."""

    count: int = Field(default=1000, ge=1, description="Target count (reviews or sentences based on count_mode)")
    count_mode: str = Field(default="reviews", description="Count mode: 'reviews' or 'sentences'")
    target_sentences: Optional[int] = Field(default=None, ge=1, description="Target sentence count (when count_mode='sentences')")
    batch_size: int = Field(default=50, ge=1, description="Reviews per batch")
    request_size: int = Field(default=5, ge=1, description="Parallel API requests")
    mode: str = Field(default="single-provider", description="Generation mode")
    provider: str = Field(default="anthropic", description="LLM provider")
    model: str = Field(default="claude-sonnet-4", description="Model name")
    dataset_mode: str = Field(default="explicit", description="Dataset annotation mode: explicit, implicit, or both")


class OutputConfig(BaseModel):
    """Output configuration."""

    formats: list[str] = Field(
        default_factory=lambda: ["jsonl"],
        description="Output formats: jsonl, csv, semeval_xml",
    )
    directory: str = Field(default="./output", description="Output directory")
    include_metadata: bool = Field(default=True, description="Include metadata in output")


class CeraConfig(BaseModel):
    """Complete CERA configuration."""

    subject_profile: SubjectProfile
    reviewer_profile: ReviewerProfile = Field(default_factory=ReviewerProfile)
    attributes_profile: AttributesProfile = Field(default_factory=AttributesProfile)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @classmethod
    def from_json_file(cls, path: str) -> "CeraConfig":
        """Load configuration from JSON file."""
        import json

        with open(path) as f:
            data = json.load(f)
        return cls(**data)

    def to_json_file(self, path: str) -> None:
        """Save configuration to JSON file."""
        import json

        with open(path, "w") as f:
            json.dump(self.model_dump(), f, indent=2)
