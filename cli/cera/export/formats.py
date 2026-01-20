"""Export formats for CERA datasets."""

import json
import csv
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET
from xml.dom import minidom


def export_jsonl(reviews: list[dict[str, Any]], output_path: Path) -> None:
    """
    Export reviews to JSONL format (JSON Lines).

    Each line is a valid JSON object containing:
    - text: The review text
    - aspects: List of aspect-sentiment pairs
    - sentiment: Overall sentiment
    - metadata: Optional metadata (reviewer profile, etc.)

    Args:
        reviews: List of review dictionaries
        output_path: Path to output file
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for review in reviews:
            # Ensure consistent structure
            export_record = {
                "text": review.get("text", ""),
                "aspects": review.get("aspects", []),
                "sentiment": review.get("sentiment", "neutral"),
            }

            # Include optional fields if present
            if "metadata" in review:
                export_record["metadata"] = review["metadata"]
            if "id" in review:
                export_record["id"] = review["id"]

            f.write(json.dumps(export_record, ensure_ascii=False) + "\n")


def export_csv(reviews: list[dict[str, Any]], output_path: Path) -> None:
    """
    Export reviews to CSV format.

    Columns:
    - id: Review ID
    - text: Review text
    - sentiment: Overall sentiment (positive/neutral/negative)
    - polarity: Numeric polarity (-1, 0, 1)
    - aspects: Semicolon-separated aspect:sentiment pairs

    Args:
        reviews: List of review dictionaries
        output_path: Path to output file
    """
    fieldnames = ["id", "text", "sentiment", "polarity", "aspects"]

    polarity_map = {"positive": 1, "neutral": 0, "negative": -1}

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()

        for i, review in enumerate(reviews):
            sentiment = review.get("sentiment", "neutral")
            aspects = review.get("aspects", [])

            # Format aspects as "aspect:sentiment" pairs
            aspect_strs = []
            for asp in aspects:
                if isinstance(asp, dict):
                    term = asp.get("term", asp.get("aspect", ""))
                    sent = asp.get("sentiment", asp.get("polarity", "neutral"))
                    aspect_strs.append(f"{term}:{sent}")
                elif isinstance(asp, str):
                    aspect_strs.append(asp)

            writer.writerow({
                "id": review.get("id", i + 1),
                "text": review.get("text", ""),
                "sentiment": sentiment,
                "polarity": polarity_map.get(sentiment, 0),
                "aspects": ";".join(aspect_strs),
            })


def export_semeval(
    reviews: list[dict[str, Any]],
    output_path: Path,
    domain: str = "restaurant"
) -> None:
    """
    Export reviews to SemEval XML format.

    This format is compatible with SemEval ABSA shared tasks (2014-2016).

    Structure:
    <sentences>
      <sentence id="1">
        <text>Review text here</text>
        <aspectTerms>
          <aspectTerm term="food" polarity="positive" from="0" to="4"/>
        </aspectTerms>
        <aspectCategories>
          <aspectCategory category="food" polarity="positive"/>
        </aspectCategories>
      </sentence>
    </sentences>

    Args:
        reviews: List of review dictionaries
        output_path: Path to output file
        domain: Domain name (e.g., "restaurant", "laptop")
    """
    root = ET.Element("sentences")
    root.set("domain", domain)

    for i, review in enumerate(reviews):
        sentence = ET.SubElement(root, "sentence")
        sentence.set("id", str(review.get("id", i + 1)))

        # Add review text
        text_elem = ET.SubElement(sentence, "text")
        text_elem.text = review.get("text", "")

        aspects = review.get("aspects", [])

        if aspects:
            # Add aspect terms
            aspect_terms = ET.SubElement(sentence, "aspectTerms")
            aspect_categories = ET.SubElement(sentence, "aspectCategories")

            review_text = review.get("text", "").lower()

            for asp in aspects:
                if isinstance(asp, dict):
                    term = asp.get("term", asp.get("aspect", ""))
                    polarity = asp.get("sentiment", asp.get("polarity", "neutral"))
                    category = asp.get("category", term)

                    # Try to find term position in text
                    term_lower = term.lower()
                    start_idx = review_text.find(term_lower)
                    if start_idx == -1:
                        start_idx = 0
                    end_idx = start_idx + len(term)

                    # Add aspect term
                    asp_term = ET.SubElement(aspect_terms, "aspectTerm")
                    asp_term.set("term", term)
                    asp_term.set("polarity", polarity)
                    asp_term.set("from", str(start_idx))
                    asp_term.set("to", str(end_idx))

                    # Add aspect category
                    asp_cat = ET.SubElement(aspect_categories, "aspectCategory")
                    asp_cat.set("category", category)
                    asp_cat.set("polarity", polarity)

    # Pretty print XML
    xml_str = ET.tostring(root, encoding="unicode")
    dom = minidom.parseString(xml_str)
    pretty_xml = dom.toprettyxml(indent="  ")

    # Remove extra blank lines
    lines = [line for line in pretty_xml.split("\n") if line.strip()]

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def load_dataset(input_path: Path) -> list[dict[str, Any]]:
    """
    Load a dataset from file (JSONL or CSV).

    Args:
        input_path: Path to input file

    Returns:
        List of review dictionaries
    """
    suffix = input_path.suffix.lower()

    if suffix == ".jsonl":
        return _load_jsonl(input_path)
    elif suffix == ".csv":
        return _load_csv(input_path)
    elif suffix == ".json":
        return _load_json(input_path)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSONL file."""
    reviews = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                reviews.append(json.loads(line))
    return reviews


def _load_csv(path: Path) -> list[dict[str, Any]]:
    """Load CSV file."""
    reviews = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            review = {
                "id": row.get("id", len(reviews) + 1),
                "text": row.get("text", ""),
                "sentiment": row.get("sentiment", "neutral"),
            }

            # Parse aspects
            aspects_str = row.get("aspects", "")
            if aspects_str:
                aspects = []
                for asp_str in aspects_str.split(";"):
                    if ":" in asp_str:
                        term, sentiment = asp_str.split(":", 1)
                        aspects.append({"term": term, "sentiment": sentiment})
                    else:
                        aspects.append({"term": asp_str, "sentiment": "neutral"})
                review["aspects"] = aspects

            reviews.append(review)
    return reviews


def _load_json(path: Path) -> list[dict[str, Any]]:
    """Load JSON file (array of reviews)."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and "reviews" in data:
            return data["reviews"]
        else:
            raise ValueError("Invalid JSON structure: expected array or object with 'reviews' key")
