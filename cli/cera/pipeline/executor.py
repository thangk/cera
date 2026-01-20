"""Pipeline Executor - Orchestrates the CERA pipeline execution."""

import asyncio
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Callable
import json
import uuid

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from cera.pipeline.composition.sil import SubjectIntelligenceLayer, SubjectContext, MAVConfig
from cera.pipeline.composition.rgm import ReviewerGenerationModule, ReviewerContext
from cera.pipeline.composition.acm import AttributesCompositionModule, AttributesContext
from cera.pipeline.generation.aml import AuthenticityModelingLayer, GeneratedReview
from cera.pipeline.generation.noise import NoiseInjector, create_noise_pipeline
from cera.pipeline.evaluation.mdqa import MultiDimensionalQualityAssessment, MDQAMetrics
from cera.models.config import CeraConfig
from cera.models.output import Review, ReviewerInfo, ReviewMetadata, AspectSentiment, Dataset, DatasetMetrics


class PipelineExecutor:
    """
    Orchestrates the CERA pipeline execution.

    Phases:
    1. Composition: SIL → RGM → ACM
    2. Generation: AML (batch processing) → Noise injection
    3. Evaluation: MDQA metrics
    """

    def __init__(
        self,
        config: CeraConfig,
        api_key: str,
        console: Optional[Console] = None,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ):
        """
        Initialize the pipeline executor.

        Args:
            config: CERA configuration
            api_key: API key for LLM provider
            console: Rich console for output (optional)
            progress_callback: Callback for progress updates (progress%, phase)
        """
        self.config = config
        self.api_key = api_key
        self.console = console or Console()
        self.progress_callback = progress_callback

        # Initialize components
        # Build MAV config if provided in subject_profile
        mav_config = None
        if hasattr(config.subject_profile, 'mav') and config.subject_profile.mav:
            mav = config.subject_profile.mav
            mav_config = MAVConfig(
                enabled=mav.enabled,
                models=mav.models,
                similarity_threshold=mav.similarity_threshold,
            )

        self.sil = SubjectIntelligenceLayer(api_key, mav_config=mav_config)
        self.rgm = ReviewerGenerationModule(
            age_range=tuple(config.reviewer_profile.age_range),
            sex_distribution={
                "male": config.reviewer_profile.sex_distribution.male,
                "female": config.reviewer_profile.sex_distribution.female,
                "unspecified": config.reviewer_profile.sex_distribution.unspecified,
            },
            audience_contexts=config.reviewer_profile.audience_context,
        )
        self.acm = AttributesCompositionModule(
            polarity={
                "positive": config.attributes_profile.polarity.positive,
                "neutral": config.attributes_profile.polarity.neutral,
                "negative": config.attributes_profile.polarity.negative,
            },
            noise={
                "typo_rate": config.attributes_profile.noise.typo_rate,
                "colloquialism": config.attributes_profile.noise.colloquialism,
                "grammar_errors": config.attributes_profile.noise.grammar_errors,
            },
            length_range=tuple(config.attributes_profile.length_range),
            temp_range=tuple(config.attributes_profile.temp_range),
        )
        self.aml = AuthenticityModelingLayer(
            api_key=api_key,
            provider=config.generation.provider,
            model=config.generation.model,
        )

        # Create noise injector
        noise_cfg = config.attributes_profile.noise
        self.noise_injector = create_noise_pipeline(
            typo_rate=noise_cfg.typo_rate,
            colloquialism=noise_cfg.colloquialism,
            grammar_errors=noise_cfg.grammar_errors,
            preset=noise_cfg.preset,
            advanced=noise_cfg.advanced,
            use_ocr=noise_cfg.use_ocr,
            use_contextual=noise_cfg.use_contextual,
        )

        self.mdqa = MultiDimensionalQualityAssessment(use_gpu=False)

    def _update_progress(self, progress: int, phase: str):
        """Update progress via callback if available."""
        if self.progress_callback:
            self.progress_callback(progress, phase)

    async def _run_composition_phase(self) -> tuple[SubjectContext, list[ReviewerContext], AttributesContext]:
        """
        Phase 1: Composition

        SIL → RGM → ACM
        """
        self._update_progress(5, "composition")

        # SIL - Subject Intelligence Layer
        self.console.print("\n[bold cyan]Phase 1: Composition[/bold cyan]")
        self.console.print("[dim]  SIL[/dim] Gathering subject intelligence...")

        subject_ctx = await self.sil.gather_intelligence(
            query=self.config.subject_profile.query,
            region=self.config.subject_profile.region,
            category=self.config.subject_profile.category,
            feature_count=self.config.subject_profile.feature_count,
            sentiment_depth=self.config.subject_profile.sentiment_depth,
            context_scope=self.config.subject_profile.context_scope,
        )
        mav_status = "[green]MAV verified[/green]" if subject_ctx.mav_verified else "[yellow]single-model[/yellow]"
        self.console.print(f"[green]  OK[/green] Subject: {subject_ctx.subject} ({mav_status})")
        self.console.print(f"[dim]     Features: {len(subject_ctx.features)} | Pros: {len(subject_ctx.pros)} | Cons: {len(subject_ctx.cons)}[/dim]")
        if subject_ctx.search_sources:
            self.console.print(f"[dim]     Sources: {len(subject_ctx.search_sources)} web pages[/dim]")

        self._update_progress(15, "composition")

        # RGM - Reviewer Generation Module
        self.console.print("[dim]  RGM[/dim] Generating reviewer profiles...")
        reviewer_contexts = self.rgm.generate_profiles(self.config.generation.count)
        self.console.print(f"[green]  OK[/green] Generated {len(reviewer_contexts)} reviewer profiles")

        # Show demographic summary
        ages = [r.age for r in reviewer_contexts]
        sexes = [r.sex for r in reviewer_contexts]
        self.console.print(f"[dim]     Age range: {min(ages)}-{max(ages)} | Male: {sexes.count('male')} | Female: {sexes.count('female')}[/dim]")

        self._update_progress(25, "composition")

        # ACM - Attributes Composition Module
        self.console.print("[dim]  ACM[/dim] Composing review attributes...")
        attrs_ctx = self.acm.compose()
        self.console.print(f"[green]  OK[/green] Polarity: {attrs_ctx.polarity_distribution.positive:.0%} pos / {attrs_ctx.polarity_distribution.neutral:.0%} neu / {attrs_ctx.polarity_distribution.negative:.0%} neg")

        return subject_ctx, reviewer_contexts, attrs_ctx

    async def _run_generation_phase(
        self,
        subject_ctx: SubjectContext,
        reviewer_contexts: list[ReviewerContext],
        attrs_ctx: AttributesContext,
    ) -> list[GeneratedReview]:
        """
        Phase 2: Generation

        AML batch processing → Noise injection
        """
        self._update_progress(30, "generation")

        self.console.print("\n[bold cyan]Phase 2: Generation[/bold cyan]")

        total_reviews = self.config.generation.count
        batch_size = self.config.generation.batch_size
        num_batches = (total_reviews + batch_size - 1) // batch_size

        all_reviews = []

        # Convert contexts to dicts for AML
        subject_dict = asdict(subject_ctx)
        attrs_dict = {
            "polarity_distribution": {
                "positive": attrs_ctx.polarity_distribution.positive,
                "neutral": attrs_ctx.polarity_distribution.neutral,
                "negative": attrs_ctx.polarity_distribution.negative,
            },
            "noise_config": asdict(attrs_ctx.noise_config),
            "temperature": attrs_ctx.temperature,
            "length_sentences": attrs_ctx.length_sentences,
        }

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self.console,
        ) as progress:
            task = progress.add_task("[dim]  AML[/dim] Generating reviews...", total=num_batches)

            for batch_num in range(num_batches):
                # Calculate batch range
                start_idx = batch_num * batch_size
                end_idx = min(start_idx + batch_size, total_reviews)
                batch_reviewers = reviewer_contexts[start_idx:end_idx]

                # Convert reviewers to dicts
                reviewer_dicts = [asdict(r) for r in batch_reviewers]

                # Generate batch
                batch_reviews = await self.aml.generate_batch(
                    subject_context=subject_dict,
                    reviewer_contexts=reviewer_dicts,
                    attributes_context=attrs_dict,
                    batch_id=batch_num,
                )

                all_reviews.extend(batch_reviews)

                # Update progress
                batch_progress = 30 + int((batch_num + 1) / num_batches * 40)
                self._update_progress(batch_progress, "generation")
                progress.update(task, advance=1)

        self.console.print(f"[green]  OK[/green] Generated {len(all_reviews)} reviews")

        # Show polarity distribution
        polarities = [r.polarity for r in all_reviews]
        pos_count = polarities.count("positive")
        neu_count = polarities.count("neutral")
        neg_count = polarities.count("negative")
        self.console.print(f"[dim]     Distribution: {pos_count} pos / {neu_count} neu / {neg_count} neg[/dim]")

        # Apply noise injection
        self._update_progress(75, "generation")
        self.console.print("[dim]  NOISE[/dim] Injecting human-like imperfections...")

        noisy_reviews, clean_reviews = self.noise_injector.inject_noise_batch(
            all_reviews, text_field="text"
        )

        self.console.print(f"[green]  OK[/green] Noise applied (typo_rate={self.config.attributes_profile.noise.typo_rate:.1%})")

        return noisy_reviews

    async def _run_evaluation_phase(self, reviews: list[GeneratedReview]) -> MDQAMetrics:
        """
        Phase 3: Evaluation

        MDQA metrics computation
        """
        self._update_progress(80, "evaluation")

        self.console.print("\n[bold cyan]Phase 3: Evaluation[/bold cyan]")
        self.console.print("[dim]  MDQA[/dim] Computing quality metrics...")

        # Extract texts for evaluation
        texts = [r.text for r in reviews]

        # Compute metrics (diversity metrics only since we don't have references)
        self._update_progress(85, "evaluation")
        metrics = self.mdqa.evaluate(texts)

        self._update_progress(95, "evaluation")

        self.console.print(f"[green]  OK[/green] Evaluation complete")
        self.console.print(f"[dim]     Distinct-1: {metrics.distinct_1:.3f} | Distinct-2: {metrics.distinct_2:.3f} | Self-BLEU: {metrics.self_bleu:.3f}[/dim]")

        return metrics

    def _save_output(
        self,
        reviews: list[GeneratedReview],
        metrics: Optional[MDQAMetrics],
    ) -> Path:
        """Save generated dataset to output directory."""
        # Create output directory
        output_dir = Path(self.config.output.directory)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate dataset ID
        dataset_id = f"{self.config.subject_profile.query.lower().replace(' ', '-')}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        # Convert GeneratedReview objects to output format
        output_reviews = []
        for i, review in enumerate(reviews):
            output_review = {
                "id": f"{dataset_id}-{i:05d}",
                "text": review.text,
                "polarity": review.polarity,
                "aspects": [{"term": a.get("term", ""), "sentiment": a.get("sentiment", "")} for a in review.aspects] if review.aspects else [],
                "reviewer": {
                    "age": review.reviewer_age,
                    "sex": review.reviewer_sex,
                    "audience_context": review.audience_context,
                },
                "metadata": {
                    "model": review.model,
                    "temperature": review.temperature,
                    "batch_id": review.batch_id,
                    "timestamp": datetime.utcnow().isoformat(),
                },
            }
            output_reviews.append(output_review)

        # Save to JSONL
        output_path = output_dir / f"{dataset_id}.jsonl"
        with open(output_path, "w", encoding="utf-8") as f:
            for review in output_reviews:
                f.write(json.dumps(review, ensure_ascii=False) + "\n")

        # Save metadata
        metadata = {
            "id": dataset_id,
            "name": f"{self.config.subject_profile.query} Reviews",
            "subject": self.config.subject_profile.query,
            "category": self.config.subject_profile.category,
            "review_count": len(reviews),
            "metrics": metrics.to_dict() if metrics else None,
            "config": {
                "subject_profile": {
                    "query": self.config.subject_profile.query,
                    "region": self.config.subject_profile.region,
                    "category": self.config.subject_profile.category,
                },
                "generation": {
                    "count": self.config.generation.count,
                    "model": self.config.generation.model,
                    "provider": self.config.generation.provider,
                },
                "noise": {
                    "typo_rate": self.config.attributes_profile.noise.typo_rate,
                    "colloquialism": self.config.attributes_profile.noise.colloquialism,
                    "grammar_errors": self.config.attributes_profile.noise.grammar_errors,
                },
            },
            "created_at": datetime.utcnow().isoformat(),
        }

        metadata_path = output_dir / f"{dataset_id}.meta.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        return output_path

    async def execute(self, skip_eval: bool = False) -> tuple[Path, Optional[MDQAMetrics]]:
        """
        Execute the full CERA pipeline.

        Args:
            skip_eval: Skip MDQA evaluation phase

        Returns:
            Tuple of (output_path, metrics) - metrics is None if skip_eval=True
        """
        # Check if using real LLM or placeholder
        is_placeholder = self.aml._is_placeholder_mode()
        mode_str = "[yellow]placeholder[/yellow]" if is_placeholder else "[green]OpenRouter[/green]"

        self.console.print("\n[bold]Starting CERA Pipeline[/bold]")
        self.console.print(f"[dim]Subject: {self.config.subject_profile.query}[/dim]")
        self.console.print(f"[dim]Count: {self.config.generation.count} reviews[/dim]")
        self.console.print(f"[dim]Model: {self.config.generation.provider}/{self.config.generation.model}[/dim]")
        self.console.print(f"[dim]Mode: {mode_str}[/dim]")

        try:
            # Phase 1: Composition
            subject_ctx, reviewer_contexts, attrs_ctx = await self._run_composition_phase()

            # Phase 2: Generation
            reviews = await self._run_generation_phase(subject_ctx, reviewer_contexts, attrs_ctx)

            # Phase 3: Evaluation (optional)
            metrics = None
            if not skip_eval:
                metrics = await self._run_evaluation_phase(reviews)
            else:
                self.console.print("\n[dim]Skipping evaluation (--skip-eval)[/dim]")
                self._update_progress(95, "skipped_eval")

            # Save output
            self._update_progress(98, "saving")
            self.console.print("\n[bold cyan]Saving Output[/bold cyan]")
            output_path = self._save_output(reviews, metrics)
            self.console.print(f"[green]  OK[/green] Saved to {output_path}")

            self._update_progress(100, "complete")

            return output_path, metrics
        finally:
            # Clean up AML client
            await self.aml.close()


async def run_pipeline(
    config: CeraConfig,
    api_key: str,
    console: Optional[Console] = None,
    progress_callback: Optional[Callable[[int, str], None]] = None,
    skip_eval: bool = False,
) -> tuple[Path, Optional[MDQAMetrics]]:
    """
    Convenience function to run the CERA pipeline.

    Args:
        config: CERA configuration
        api_key: API key for LLM provider
        console: Rich console for output
        progress_callback: Progress callback function
        skip_eval: Skip MDQA evaluation phase

    Returns:
        Tuple of (output_path, metrics) - metrics is None if skip_eval=True
    """
    executor = PipelineExecutor(
        config=config,
        api_key=api_key,
        console=console,
        progress_callback=progress_callback,
    )
    return await executor.execute(skip_eval=skip_eval)
