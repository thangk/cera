"""CERA CLI - Command-line interface for CERA pipeline."""

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from pathlib import Path
from typing import Optional
import json
import asyncio
import os

app = typer.Typer(
    name="cera",
    help="CERA - Context-Engineered Reviews Architecture for synthetic ABSA dataset generation",
    add_completion=False,
)
console = Console()


@app.command()
def run(
    config: Path = typer.Argument(
        ...,
        help="Path to the configuration JSON file",
        exists=True,
        readable=True,
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output", "-o",
        help="Output name/directory (optional, auto-generated from config if not provided)",
    ),
    count: Optional[int] = typer.Option(
        None,
        "--count", "-n",
        help="Number of reviews to generate (overrides config)",
    ),
    typo_rate: Optional[float] = typer.Option(
        None,
        "--typo-rate",
        help="Character-level typo rate (0.0-0.1, e.g., 0.01 = 1%)",
        min=0.0,
        max=0.1,
    ),
    colloquialism: Optional[bool] = typer.Option(
        None,
        "--colloquialism/--no-colloquialism",
        help="Enable/disable colloquial word substitutions",
    ),
    grammar_errors: Optional[bool] = typer.Option(
        None,
        "--grammar-errors/--no-grammar-errors",
        help="Enable/disable minor grammar imperfections",
    ),
    noise_preset: Optional[str] = typer.Option(
        None,
        "--noise-preset",
        help="Noise preset: 'none', 'light', 'moderate', 'heavy'",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate config without running pipeline",
    ),
    skip_eval: bool = typer.Option(
        False,
        "--skip-eval",
        help="Skip MDQA evaluation after generation",
    ),
):
    """
    Run the CERA pipeline to generate synthetic ABSA reviews.

    \b
    Examples:
        cera run config.json
        cera run config.json --output restaurant-reviews-2000
        cera run config.json -n 500 --noise-preset moderate
        cera run config.json --dry-run
    """
    console.print(Panel.fit(
        "[bold blue]CERA[/bold blue] - Context-Engineered Reviews Architecture",
        subtitle="Synthetic ABSA Dataset Generation",
    ))

    # Load and validate config
    console.print(f"\n[dim]Config:[/dim] {config}")
    with open(config) as f:
        config_data = json.load(f)

    # Override with CLI options
    if output:
        config_data.setdefault("output", {})["directory"] = output
    if count:
        config_data.setdefault("generation", {})["count"] = count

    # Apply noise preset if specified
    noise_presets = {
        "none": {"typo_rate": 0.0, "colloquialism": False, "grammar_errors": False},
        "light": {"typo_rate": 0.005, "colloquialism": False, "grammar_errors": True},
        "moderate": {"typo_rate": 0.01, "colloquialism": True, "grammar_errors": True},
        "heavy": {"typo_rate": 0.03, "colloquialism": True, "grammar_errors": True},
    }

    if noise_preset:
        if noise_preset not in noise_presets:
            console.print(f"[red]Error:[/red] Invalid noise preset '{noise_preset}'")
            console.print(f"[dim]Valid presets: {', '.join(noise_presets.keys())}[/dim]")
            raise typer.Exit(1)
        preset = noise_presets[noise_preset]
        config_data.setdefault("attributes_profile", {}).setdefault("noise", {})
        config_data["attributes_profile"]["noise"].update(preset)
        console.print(f"[dim]Noise preset:[/dim] {noise_preset}")

    # Override individual noise settings (these take precedence over presets)
    if typo_rate is not None:
        config_data.setdefault("attributes_profile", {}).setdefault("noise", {})["typo_rate"] = typo_rate
    if colloquialism is not None:
        config_data.setdefault("attributes_profile", {}).setdefault("noise", {})["colloquialism"] = colloquialism
    if grammar_errors is not None:
        config_data.setdefault("attributes_profile", {}).setdefault("noise", {})["grammar_errors"] = grammar_errors

    # Display configuration summary
    subject = config_data.get("subject_profile", {}).get("query", "Unknown")
    gen_count = config_data.get("generation", {}).get("count", 1000)
    output_dir = config_data.get("output", {}).get("directory", "./output")

    console.print(f"[dim]Subject:[/dim] {subject}")
    console.print(f"[dim]Count:[/dim] {gen_count} reviews")
    console.print(f"[dim]Output:[/dim] {output_dir}")

    # Display noise configuration
    noise_cfg = config_data.get("attributes_profile", {}).get("noise", {})
    if noise_cfg:
        console.print(f"[dim]Noise:[/dim] typo={noise_cfg.get('typo_rate', 0.01):.1%}, "
                      f"colloquial={noise_cfg.get('colloquialism', True)}, "
                      f"grammar={noise_cfg.get('grammar_errors', True)}")

    console.print("[green]OK[/green] Configuration loaded")

    if dry_run:
        console.print("\n[yellow]Dry run mode[/yellow] - Config validation only")
        console.print(json.dumps(config_data, indent=2))
        return

    # Get API key from environment or prompt
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        console.print("\n[yellow]Warning:[/yellow] No API key found in environment")
        console.print("[dim]Set OPENROUTER_API_KEY or OPENAI_API_KEY environment variable[/dim]")
        console.print("[dim]Using placeholder mode for testing...[/dim]")
        api_key = "placeholder"

    # Load config using CeraConfig model
    try:
        from cera.models.config import CeraConfig
        cera_config = CeraConfig(**config_data)
    except Exception as e:
        console.print(f"[red]Error parsing config:[/red] {e}")
        raise typer.Exit(1)

    # Run the pipeline
    try:
        from cera.pipeline.executor import run_pipeline

        output_path, metrics = asyncio.run(
            run_pipeline(
                config=cera_config,
                api_key=api_key,
                console=console,
                skip_eval=skip_eval,
            )
        )

        # Show final results
        console.print("\n")
        console.print(Panel.fit(
            "[bold green]Pipeline Complete![/bold green]",
            subtitle="CERA Synthetic ABSA Dataset Generation",
        ))

        # Results table
        table = Table(show_header=True, header_style="bold")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")

        table.add_row("Output", str(output_path))
        table.add_row("Reviews", str(cera_config.generation.count))

        if metrics:
            table.add_row("Distinct-1", f"{metrics.distinct_1:.3f}" if metrics.distinct_1 else "N/A")
            table.add_row("Distinct-2", f"{metrics.distinct_2:.3f}" if metrics.distinct_2 else "N/A")
            table.add_row("Self-BLEU", f"{metrics.self_bleu:.3f}" if metrics.self_bleu else "N/A")
        else:
            table.add_row("Evaluation", "[dim]Skipped[/dim]")

        console.print(table)

    except KeyboardInterrupt:
        console.print("\n[yellow]Pipeline interrupted by user[/yellow]")
        raise typer.Exit(130)
    except Exception as e:
        console.print(f"\n[red]Pipeline error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind to"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload for development"),
):
    """Start the FastAPI server for web GUI integration."""
    import uvicorn

    console.print(Panel.fit(
        "[bold blue]CERA[/bold blue] API Server",
        subtitle=f"Running on http://{host}:{port}",
    ))

    uvicorn.run(
        "cera.api:app",
        host=host,
        port=port,
        reload=reload,
    )


@app.command()
def evaluate(
    dataset: Path = typer.Argument(
        ...,
        help="Path to the generated dataset (JSONL or CSV)",
        exists=True,
        readable=True,
    ),
    reference: Optional[Path] = typer.Option(
        None,
        "--reference", "-r",
        help="Path to reference dataset for comparison",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Path to save evaluation report",
    ),
    use_gpu: bool = typer.Option(
        False,
        "--gpu",
        help="Use GPU for BERTScore/MoverScore (faster but requires CUDA)",
    ),
):
    """Evaluate a generated dataset using MDQA metrics."""
    console.print(Panel.fit(
        "[bold blue]CERA[/bold blue] - Multi-Dimensional Quality Assessment",
        subtitle="Dataset Evaluation",
    ))

    console.print(f"\n[dim]Dataset:[/dim] {dataset}")
    if reference:
        console.print(f"[dim]Reference:[/dim] {reference}")
    console.print(f"[dim]Device:[/dim] {'GPU' if use_gpu else 'CPU'}")

    # Load dataset
    from cera.export.formats import load_dataset
    from cera.pipeline.evaluation.mdqa import MultiDimensionalQualityAssessment

    try:
        reviews = load_dataset(dataset)
        console.print(f"[green]OK[/green] Loaded {len(reviews)} reviews")
    except Exception as e:
        console.print(f"[red]Error loading dataset:[/red] {e}")
        raise typer.Exit(1)

    # Extract texts
    if isinstance(reviews[0], dict):
        texts = [r.get("text", "") for r in reviews]
    else:
        texts = [r.text for r in reviews]

    # Load reference texts if provided
    reference_texts = None
    if reference:
        try:
            ref_reviews = load_dataset(reference)
            if isinstance(ref_reviews[0], dict):
                reference_texts = [r.get("text", "") for r in ref_reviews]
            else:
                reference_texts = [r.text for r in ref_reviews]
            console.print(f"[green]OK[/green] Loaded {len(reference_texts)} reference reviews")
        except Exception as e:
            console.print(f"[yellow]Warning:[/yellow] Could not load reference: {e}")

    # Run MDQA evaluation
    console.print("\n[bold cyan]Running MDQA Evaluation[/bold cyan]")
    mdqa = MultiDimensionalQualityAssessment(use_gpu=use_gpu)

    with console.status("[dim]Computing metrics...[/dim]"):
        metrics = mdqa.evaluate(texts, reference_texts)

    console.print("[green]OK[/green] Evaluation complete\n")

    # Display results
    table = Table(title="MDQA Metrics", show_header=True, header_style="bold")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    table.add_column("Description", style="dim")

    # Corpus Diversity metrics (always available)
    table.add_row(
        "Distinct-1",
        f"{metrics.distinct_1:.4f}" if metrics.distinct_1 is not None else "N/A",
        "Unigram diversity (higher = more diverse)"
    )
    table.add_row(
        "Distinct-2",
        f"{metrics.distinct_2:.4f}" if metrics.distinct_2 is not None else "N/A",
        "Bigram diversity (higher = more diverse)"
    )
    table.add_row(
        "Self-BLEU",
        f"{metrics.self_bleu:.4f}" if metrics.self_bleu is not None else "N/A",
        "Internal similarity (lower = more diverse)"
    )

    # Reference-based metrics (if reference provided)
    if reference_texts:
        table.add_row("", "", "")  # Separator
        table.add_row(
            "BLEU",
            f"{metrics.bleu:.4f}" if metrics.bleu is not None else "N/A",
            "N-gram precision vs reference"
        )
        table.add_row(
            "ROUGE-L",
            f"{metrics.rouge_l:.4f}" if metrics.rouge_l is not None else "N/A",
            "Longest common subsequence"
        )
        table.add_row(
            "BERTScore",
            f"{metrics.bertscore:.4f}" if metrics.bertscore is not None else "N/A",
            "Semantic similarity (BERT embeddings)"
        )
        table.add_row(
            "MoverScore",
            f"{metrics.moverscore:.4f}" if metrics.moverscore is not None else "N/A",
            "Word mover distance similarity"
        )

    console.print(table)

    # Overall score
    overall = metrics.overall_score
    console.print(f"\n[bold]Overall Quality Score:[/bold] {overall:.3f}")

    # Interpretation
    if overall >= 0.7:
        console.print("[green]Excellent[/green] - High quality and diversity")
    elif overall >= 0.5:
        console.print("[yellow]Good[/yellow] - Acceptable quality")
    else:
        console.print("[red]Needs improvement[/red] - Consider adjusting generation parameters")

    # Polarity distribution (if available)
    if isinstance(reviews[0], dict) and "polarity" in reviews[0]:
        polarities = [r.get("polarity", "unknown") for r in reviews]
        pos = polarities.count("positive")
        neu = polarities.count("neutral")
        neg = polarities.count("negative")
        total = len(polarities)

        console.print(f"\n[bold]Polarity Distribution:[/bold]")
        console.print(f"  Positive: {pos} ({pos/total:.1%})")
        console.print(f"  Neutral:  {neu} ({neu/total:.1%})")
        console.print(f"  Negative: {neg} ({neg/total:.1%})")

    # Save report if output specified
    if output:
        report = {
            "dataset": str(dataset),
            "reference": str(reference) if reference else None,
            "review_count": len(reviews),
            "metrics": metrics.to_dict(),
            "overall_score": overall,
            "device": "GPU" if use_gpu else "CPU",
        }

        # Add polarity distribution if available
        if isinstance(reviews[0], dict) and "polarity" in reviews[0]:
            polarities = [r.get("polarity", "unknown") for r in reviews]
            report["polarity_distribution"] = {
                "positive": polarities.count("positive") / len(polarities),
                "neutral": polarities.count("neutral") / len(polarities),
                "negative": polarities.count("negative") / len(polarities),
            }

        with open(output, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        console.print(f"\n[green]OK[/green] Report saved to {output}")


@app.command()
def export(
    dataset: Path = typer.Argument(
        ...,
        help="Path to the generated dataset",
        exists=True,
        readable=True,
    ),
    format: str = typer.Option(
        "jsonl",
        "--format", "-f",
        help="Output format: jsonl, csv, or semeval",
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output file path",
    ),
    domain: str = typer.Option(
        "restaurant",
        "--domain", "-d",
        help="Domain name for SemEval format",
    ),
):
    """Export a dataset to different formats."""
    from cera.export import DatasetExporter
    from cera.export.formats import load_dataset

    console.print(Panel.fit(
        "[bold blue]CERA[/bold blue] - Dataset Export",
        subtitle=f"Format: {format.upper()}",
    ))

    # Validate format
    valid_formats = ["jsonl", "csv", "semeval"]
    if format not in valid_formats:
        console.print(f"[red]Error:[/red] Invalid format '{format}'")
        console.print(f"[dim]Valid formats: {', '.join(valid_formats)}[/dim]")
        raise typer.Exit(1)

    console.print(f"\n[dim]Input:[/dim] {dataset}")
    console.print(f"[dim]Format:[/dim] {format}")

    # Load dataset
    try:
        reviews = load_dataset(dataset)
        console.print(f"[green]OK[/green] Loaded {len(reviews)} reviews")
    except Exception as e:
        console.print(f"[red]Error loading dataset:[/red] {e}")
        raise typer.Exit(1)

    # Determine output path
    if output is None:
        ext_map = {"jsonl": ".jsonl", "csv": ".csv", "semeval": ".xml"}
        output = dataset.with_suffix(ext_map[format])

    console.print(f"[dim]Output:[/dim] {output}")

    # Export
    try:
        exporter = DatasetExporter(domain=domain)
        result_path = exporter.export(reviews, output, format)

        # Show stats
        stats = exporter.get_stats(reviews)
        console.print(f"\n[green]OK[/green] Exported to {result_path}")
        console.print(f"[dim]  Reviews: {stats['total_reviews']}[/dim]")
        console.print(f"[dim]  Aspects: {stats['total_aspects']}[/dim]")
        console.print(f"[dim]  Avg words/review: {stats['avg_words_per_review']:.1f}[/dim]")
    except Exception as e:
        console.print(f"[red]Error exporting:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def version():
    """Show CERA version information."""
    from cera import __version__

    console.print(f"CERA version [bold]{__version__}[/bold]")


if __name__ == "__main__":
    app()
