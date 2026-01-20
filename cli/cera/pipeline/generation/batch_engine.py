"""Batch Engine - Parallel LLM batch processing."""

import asyncio
from dataclasses import dataclass
from typing import Callable, Optional
from .aml import GeneratedReview


@dataclass
class BatchResult:
    """Result of a batch processing operation."""

    batch_id: int
    reviews: list[GeneratedReview]
    success: bool
    error: Optional[str] = None


class BatchEngine:
    """
    Batch Engine for parallel LLM processing.

    Handles batch processing of review generation requests
    with configurable parallelism and rate limiting.
    """

    def __init__(
        self,
        batch_size: int = 50,
        request_size: int = 5,
        max_concurrent: int = 10,
    ):
        self.batch_size = batch_size
        self.request_size = request_size
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def process_batch(
        self,
        batch_id: int,
        generate_fn: Callable,
        subject_context: dict,
        reviewer_contexts: list[dict],
        attributes_context: dict,
    ) -> BatchResult:
        """
        Process a single batch of reviews.

        Args:
            batch_id: Unique batch identifier
            generate_fn: Async function to generate reviews
            subject_context: Subject intelligence data
            reviewer_contexts: List of reviewer profiles
            attributes_context: Generation attributes

        Returns:
            BatchResult with generated reviews or error
        """
        async with self.semaphore:
            try:
                reviews = await generate_fn(
                    subject_context,
                    reviewer_contexts,
                    attributes_context,
                    batch_id,
                )
                return BatchResult(
                    batch_id=batch_id,
                    reviews=reviews,
                    success=True,
                )
            except Exception as e:
                return BatchResult(
                    batch_id=batch_id,
                    reviews=[],
                    success=False,
                    error=str(e),
                )

    async def process_all(
        self,
        total_count: int,
        generate_fn: Callable,
        subject_context: dict,
        reviewer_generator: Callable,
        attributes_context: dict,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> list[GeneratedReview]:
        """
        Process all reviews in batches.

        Args:
            total_count: Total number of reviews to generate
            generate_fn: Async function to generate a batch
            subject_context: Subject intelligence data
            reviewer_generator: Function to generate reviewer profiles
            attributes_context: Generation attributes
            progress_callback: Optional callback for progress updates

        Returns:
            List of all generated reviews
        """
        num_batches = (total_count + self.batch_size - 1) // self.batch_size
        all_reviews = []

        tasks = []
        for i in range(num_batches):
            batch_count = min(self.batch_size, total_count - i * self.batch_size)
            reviewer_contexts = reviewer_generator(batch_count)

            task = self.process_batch(
                batch_id=i,
                generate_fn=generate_fn,
                subject_context=subject_context,
                reviewer_contexts=reviewer_contexts,
                attributes_context=attributes_context,
            )
            tasks.append(task)

        # Process with progress tracking
        for i, task in enumerate(asyncio.as_completed(tasks)):
            result = await task
            if result.success:
                all_reviews.extend(result.reviews)
            else:
                print(f"Batch {result.batch_id} failed: {result.error}")

            if progress_callback:
                progress_callback(i + 1, num_batches)

        return all_reviews
