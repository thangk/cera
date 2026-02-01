"""Output Models - Data structures for generated reviews and datasets."""

from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class ReviewMetadata(BaseModel):
    """Metadata for a generated review."""

    model: str = Field(..., description="Model used for generation")
    temperature: float = Field(..., description="Temperature setting")
    batch_id: int = Field(..., description="Batch identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ReviewerInfo(BaseModel):
    """Reviewer information for a generated review."""

    age: int
    sex: str
    additional_context: Optional[str] = None


class AspectSentiment(BaseModel):
    """Aspect-sentiment pair."""

    term: str = Field(..., description="Aspect term")
    sentiment: str = Field(..., description="Sentiment: positive, neutral, negative")
    confidence: float = Field(default=1.0, ge=0, le=1)


class Review(BaseModel):
    """A generated synthetic review."""

    id: str = Field(..., description="Unique review identifier")
    text: str = Field(..., description="Review text")
    polarity: str = Field(..., description="Overall sentiment polarity")
    aspects: list[AspectSentiment] = Field(
        default_factory=list, description="Extracted aspects"
    )
    reviewer: ReviewerInfo
    metadata: ReviewMetadata

    def to_semeval_xml(self) -> str:
        """Convert to SemEval XML format."""
        aspects_xml = ""
        for asp in self.aspects:
            aspects_xml += f'    <aspectTerm term="{asp.term}" polarity="{asp.sentiment}" />\n'

        return f"""<sentence id="{self.id}">
  <text>{self.text}</text>
  <aspectTerms>
{aspects_xml}  </aspectTerms>
</sentence>"""


class DatasetMetrics(BaseModel):
    """Quality metrics for a dataset."""

    bertscore: Optional[float] = None
    distinct_1: Optional[float] = None
    distinct_2: Optional[float] = None
    self_bleu: Optional[float] = None
    bleu: Optional[float] = None
    rouge_l: Optional[float] = None


class Dataset(BaseModel):
    """A generated dataset of reviews."""

    id: str = Field(..., description="Dataset identifier")
    name: str = Field(..., description="Dataset name")
    subject: str = Field(..., description="Review subject")
    category: str = Field(..., description="Subject category")
    reviews: list[Review] = Field(default_factory=list)
    metrics: Optional[DatasetMetrics] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def review_count(self) -> int:
        """Get total review count."""
        return len(self.reviews)

    def to_jsonl(self) -> str:
        """Export to JSONL format."""
        import json

        lines = [json.dumps(r.model_dump(), default=str) for r in self.reviews]
        return "\n".join(lines)

    def to_csv(self) -> str:
        """Export to CSV format."""
        import csv
        from io import StringIO

        output = StringIO()
        if not self.reviews:
            return ""

        fieldnames = ["id", "text", "polarity", "reviewer_age", "reviewer_sex"]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()

        for review in self.reviews:
            writer.writerow(
                {
                    "id": review.id,
                    "text": review.text,
                    "polarity": review.polarity,
                    "reviewer_age": review.reviewer.age,
                    "reviewer_sex": review.reviewer.sex,
                }
            )

        return output.getvalue()

    def to_semeval_xml(self) -> str:
        """Export to SemEval XML format."""
        reviews_xml = "\n".join(r.to_semeval_xml() for r in self.reviews)
        return f"""<?xml version="1.0" encoding="UTF-8"?>
<sentences>
{reviews_xml}
</sentences>"""


class EvaluationReport(BaseModel):
    """Evaluation report for a dataset."""

    dataset_id: str
    dataset_name: str
    metrics: DatasetMetrics
    polarity_distribution: dict[str, float]
    review_count: int
    created_at: datetime = Field(default_factory=datetime.utcnow)

    def summary(self) -> str:
        """Generate a text summary of the evaluation."""
        lines = [
            f"Dataset: {self.dataset_name}",
            f"Reviews: {self.review_count}",
            "",
            "Metrics:",
        ]

        if self.metrics.bertscore is not None:
            lines.append(f"  BERTScore: {self.metrics.bertscore:.3f}")
        if self.metrics.distinct_1 is not None:
            lines.append(f"  Distinct-1: {self.metrics.distinct_1:.3f}")
        if self.metrics.distinct_2 is not None:
            lines.append(f"  Distinct-2: {self.metrics.distinct_2:.3f}")
        if self.metrics.self_bleu is not None:
            lines.append(f"  Self-BLEU: {self.metrics.self_bleu:.3f}")

        lines.append("")
        lines.append("Polarity Distribution:")
        for pol, pct in self.polarity_distribution.items():
            lines.append(f"  {pol}: {pct:.1%}")

        return "\n".join(lines)
