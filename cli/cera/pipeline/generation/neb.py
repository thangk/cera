"""NEB (Negative Example Buffer) - Prevents generating similar reviews across batches.

This module maintains a rolling buffer of previously generated reviews that are
included as "negative examples" in subsequent generation prompts, instructing
the LLM to generate distinctly different reviews.
"""

from typing import Optional


class NegativeExampleBuffer:
    """
    Maintains a rolling buffer of previously generated reviews
    to prevent repetition in subsequent generation batches.

    Usage:
        neb = NegativeExampleBuffer(max_size=75)  # 3 batches * 25 request_size

        # After first batch completes
        neb.add_batch(["Review 1...", "Review 2...", ...])

        # Before generating next batch
        context = neb.get_formatted_context()  # Include in system prompt
    """

    def __init__(self, max_size: int = 25):
        """
        Initialize the NEB buffer.

        Args:
            max_size: Maximum number of reviews to keep (FIFO eviction when exceeded).
                      Typically: neb_depth * request_size
        """
        self.buffer: list[str] = []
        self.max_size = max_size

    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)

    def __bool__(self) -> bool:
        """Buffer object is always truthy when it exists (even if empty)."""
        return True

    def add_batch(self, review_texts: list[str]) -> None:
        """
        Add a batch of reviews to the buffer.

        Maintains max size using FIFO eviction (oldest reviews removed first).

        Args:
            review_texts: List of review text strings from the completed batch
        """
        self.buffer.extend(review_texts)
        # Trim oldest if over capacity
        if len(self.buffer) > self.max_size:
            self.buffer = self.buffer[-self.max_size:]

    def get_formatted_context(self, max_chars_per_review: int = 300) -> str:
        """
        Format buffer as well-structured markdown for prompt injection.

        Returns empty string if buffer is empty (first batch).

        Args:
            max_chars_per_review: Truncate reviews longer than this to save tokens

        Returns:
            Formatted markdown string for inclusion in system prompt
        """
        if not self.buffer:
            return ""

        lines = [
            "> **CRITICAL INSTRUCTION**: The reviews below have already been generated.",
            "> Your review MUST be **completely different** from ALL of these.",
            "",
            "### What to Avoid",
            "- Same opening phrases or sentence starters",
            "- Similar vocabulary and word choices",
            "- Comparable writing style and tone",
            "- Repeated specific details or examples",
            "- Similar structure and flow",
            "",
            "---",
            "",
            f"### Previously Generated Reviews ({len(self.buffer)} reviews - DO NOT IMITATE)",
            "",
        ]

        for i, review in enumerate(self.buffer, 1):
            # Truncate very long reviews for prompt efficiency
            if len(review) > max_chars_per_review:
                truncated = review[:max_chars_per_review] + "..."
            else:
                truncated = review
            # Escape any blockquote characters in the review text
            truncated = truncated.replace("\n", " ").strip()
            lines.append(f"**Review {i}:**")
            lines.append(f"> {truncated}")
            lines.append("")

        lines.extend([
            "---",
            "",
            "> **Your review must start differently and use distinct vocabulary.**",
        ])

        return "\n".join(lines)

    def clear(self) -> None:
        """Clear buffer for new generation run."""
        self.buffer = []

    def get_stats(self) -> dict:
        """Get buffer statistics for logging."""
        return {
            "current_size": len(self.buffer),
            "max_size": self.max_size,
            "utilization": len(self.buffer) / self.max_size if self.max_size > 0 else 0,
        }
