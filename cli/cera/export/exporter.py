"""Dataset exporter for CERA."""

from pathlib import Path
from typing import Any, Literal

from .formats import export_jsonl, export_csv, export_semeval, load_dataset


ExportFormat = Literal["jsonl", "csv", "semeval"]


class DatasetExporter:
    """
    High-level dataset exporter.

    Supports exporting datasets to multiple formats:
    - JSONL: JSON Lines format (one JSON object per line)
    - CSV: Comma-separated values
    - SemEval: XML format compatible with SemEval ABSA tasks

    Example:
        exporter = DatasetExporter()

        # Export from reviews in memory
        exporter.export(reviews, "output.jsonl", format="jsonl")

        # Convert between formats
        exporter.convert("input.jsonl", "output.csv", format="csv")

        # Export to all formats at once
        exporter.export_all(reviews, "output_dir/dataset")
    """

    def __init__(self, domain: str = "restaurant"):
        """
        Initialize exporter.

        Args:
            domain: Domain name for SemEval format
        """
        self.domain = domain

    def export(
        self,
        reviews: list[dict[str, Any]],
        output_path: str | Path,
        format: ExportFormat = "jsonl",
    ) -> Path:
        """
        Export reviews to specified format.

        Args:
            reviews: List of review dictionaries
            output_path: Path to output file
            format: Export format (jsonl, csv, semeval)

        Returns:
            Path to exported file
        """
        output_path = Path(output_path)

        # Ensure correct extension
        expected_ext = {
            "jsonl": ".jsonl",
            "csv": ".csv",
            "semeval": ".xml",
        }

        if output_path.suffix != expected_ext[format]:
            output_path = output_path.with_suffix(expected_ext[format])

        # Create parent directory if needed
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "jsonl":
            export_jsonl(reviews, output_path)
        elif format == "csv":
            export_csv(reviews, output_path)
        elif format == "semeval":
            export_semeval(reviews, output_path, domain=self.domain)
        else:
            raise ValueError(f"Unsupported format: {format}")

        return output_path

    def convert(
        self,
        input_path: str | Path,
        output_path: str | Path,
        format: ExportFormat = "jsonl",
    ) -> Path:
        """
        Convert a dataset from one format to another.

        Args:
            input_path: Path to input file
            output_path: Path to output file
            format: Target export format

        Returns:
            Path to converted file
        """
        input_path = Path(input_path)
        reviews = load_dataset(input_path)
        return self.export(reviews, output_path, format)

    def export_all(
        self,
        reviews: list[dict[str, Any]],
        output_base: str | Path,
    ) -> dict[str, Path]:
        """
        Export reviews to all supported formats.

        Args:
            reviews: List of review dictionaries
            output_base: Base path for output files (without extension)

        Returns:
            Dictionary mapping format to output path
        """
        output_base = Path(output_base)
        results = {}

        for format in ["jsonl", "csv", "semeval"]:
            output_path = self.export(reviews, output_base, format)
            results[format] = output_path

        return results

    def get_stats(self, reviews: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Get statistics about a dataset.

        Args:
            reviews: List of review dictionaries

        Returns:
            Dictionary with dataset statistics
        """
        total = len(reviews)

        sentiment_counts = {"positive": 0, "neutral": 0, "negative": 0}
        aspect_count = 0
        word_count = 0

        for review in reviews:
            sentiment = review.get("sentiment", "neutral")
            sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1

            aspects = review.get("aspects", [])
            aspect_count += len(aspects)

            text = review.get("text", "")
            word_count += len(text.split())

        return {
            "total_reviews": total,
            "sentiment_distribution": sentiment_counts,
            "total_aspects": aspect_count,
            "avg_aspects_per_review": aspect_count / total if total > 0 else 0,
            "avg_words_per_review": word_count / total if total > 0 else 0,
        }
