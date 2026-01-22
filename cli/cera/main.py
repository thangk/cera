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

    # Get API key from .env first, then Convex fallback
    from cera.config_loader import get_openrouter_api_key
    api_key = get_openrouter_api_key()
    if not api_key:
        console.print("\n[yellow]Warning:[/yellow] No API key found")
        console.print("[dim]Set OPENROUTER_API_KEY in .env or configure in web settings[/dim]")
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


def _fetch_openrouter_models() -> list[dict]:
    """Fetch and process models from OpenRouter API."""
    import httpx
    from cera.config_loader import get_openrouter_api_key

    api_key = get_openrouter_api_key()

    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    response = httpx.get(
        "https://openrouter.ai/api/v1/models",
        headers=headers,
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()

    models_list = data.get("data", [])
    processed_models = []

    for model in models_list:
        model_id = model.get("id", "")
        model_name = model.get("name", "")
        pricing = model.get("pricing", {})
        architecture = model.get("architecture", {})

        # Check if free
        is_free = (
            float(pricing.get("prompt", "1")) == 0 and
            float(pricing.get("completion", "1")) == 0
        )

        # Check modality (vision support)
        modality = architecture.get("modality", "text->text")
        has_vision = "image" in modality.lower() or "vision" in model_name.lower()

        # Check if open source
        is_oss = architecture.get("instruct_type") in ["llama", "mistral", "chatml", "alpaca", "vicuna", "zephyr"]
        oss_indicators = ["llama", "mistral", "mixtral", "qwen", "yi", "deepseek", "phi", "gemma", "codellama", "falcon", "mpt", "dolly", "vicuna", "openchat", "nous", "hermes"]
        if any(ind in model_id.lower() for ind in oss_indicators):
            is_oss = True

        # Check for tool support
        has_tools = architecture.get("instruct_type") in ["tool_use", "function"] or "tool" in model_name.lower() or "function" in model_name.lower()

        # Extract context length
        context_length = model.get("context_length", 0)

        # Calculate cost per 1M tokens
        prompt_cost = float(pricing.get("prompt", "0")) * 1_000_000
        completion_cost = float(pricing.get("completion", "0")) * 1_000_000

        # Get additional info
        description = model.get("description", "")
        top_provider = model.get("top_provider", {})
        max_completion = top_provider.get("max_completion_tokens", architecture.get("max_completion_tokens", 0))

        # Extract provider from model ID
        provider = model_id.split("/")[0] if "/" in model_id else "unknown"

        processed_models.append({
            "id": model_id,
            "name": model_name,
            "provider": provider,
            "context_length": context_length,
            "max_completion": max_completion,
            "prompt_cost": prompt_cost,
            "completion_cost": completion_cost,
            "is_free": is_free,
            "is_oss": is_oss,
            "has_vision": has_vision,
            "has_tools": has_tools,
            "modality": modality,
            "description": description[:100] + "..." if len(description) > 100 else description,
        })

    return processed_models


def _format_model_context(context_length: int) -> str:
    """Format context length for display."""
    if not context_length:
        return "?"
    if context_length >= 1_000_000:
        return f"{context_length / 1_000_000:.1f}M"
    if context_length >= 1000:
        return f"{context_length // 1000}K"
    return str(context_length)


def _format_model_pricing(model: dict) -> str:
    """Format model pricing for display."""
    if model["is_free"]:
        return "[green]Free[/green]"

    def fmt_cost(cost: float) -> str:
        if cost == 0:
            return "0"
        if cost < 0.01:
            return f"{cost:.3f}"
        if cost < 1:
            return f"{cost:.2f}"
        return f"{cost:.1f}"

    return f"${fmt_cost(model['prompt_cost'])}/{fmt_cost(model['completion_cost'])}"


def _supports_online_search(model_id: str) -> bool:
    """Check if model supports :online suffix for web search."""
    # Perplexity models have built-in search (always on)
    if model_id.startswith("perplexity/"):
        return True
    # These providers support :online suffix via OpenRouter
    online_providers = ["anthropic", "openai", "google"]
    provider = model_id.split("/")[0] if "/" in model_id else ""
    if provider in online_providers:
        return True
    # Specific search models
    search_models = ["openai/gpt-4o-search-preview", "openai/gpt-4o-mini-search-preview"]
    if model_id in search_models:
        return True
    return False


def _build_model_badges(model: dict) -> str:
    """Build badges string for model."""
    badges = []
    if model["is_free"]:
        badges.append("[green]Free[/green]")
    if model["is_oss"]:
        badges.append("[blue]OSS[/blue]")
    if model.get("has_online", False):
        badges.append("[cyan]Online[/cyan]")
    if model["has_vision"]:
        badges.append("[magenta]Vision[/magenta]")
    if model["has_tools"]:
        badges.append("[yellow]Tools[/yellow]")
    return " ".join(badges) if badges else "[dim]-[/dim]"


def _display_models_table(models: list[dict], detailed: bool = False):
    """Display a detailed table of models after provider selection."""
    table = Table(show_header=True, header_style="bold", show_lines=True)
    table.add_column("Name", style="white", no_wrap=False, max_width=30)
    table.add_column("Model ID (for config)", style="cyan", no_wrap=True)
    table.add_column("Context", justify="right")
    table.add_column("Price ($/1M)\nIn / Out", justify="right")
    table.add_column("Modality", max_width=20)
    table.add_column("Badges", justify="left")

    for model in models:
        # Add online capability check
        model["has_online"] = _supports_online_search(model["id"])

        # Format pricing
        pricing = _format_model_pricing(model)

        # Format modality more readably
        modality = model.get("modality", "text→text")
        modality = modality.replace("->", "→").replace("text", "txt").replace("image", "img")
        if len(modality) > 20:
            modality = modality[:17] + "..."

        table.add_row(
            model["name"],
            model["id"],
            _format_model_context(model["context_length"]),
            pricing,
            modality,
            _build_model_badges(model),
        )

    console.print(table)

    # Legend
    console.print("\n[bold]Badge Legend:[/bold]")
    console.print("  [green]Free[/green] = No cost    [blue]OSS[/blue] = Open Source    [cyan]Online[/cyan] = Web Search (:online)")
    console.print("  [magenta]Vision[/magenta] = Image input    [yellow]Tools[/yellow] = Function calling")


@app.command()
def models(
    search: Optional[str] = typer.Argument(
        None,
        help="Filter models by name (case-insensitive)",
    ),
    provider: Optional[str] = typer.Option(
        None,
        "--provider", "-p",
        help="Filter by provider (e.g., anthropic, openai, google)",
    ),
    free_only: bool = typer.Option(
        False,
        "--free",
        help="Show only free models",
    ),
    vision_only: bool = typer.Option(
        False,
        "--vision",
        help="Show only models with vision/image support",
    ),
    limit: int = typer.Option(
        50,
        "--limit", "-l",
        help="Maximum number of models to show",
    ),
    json_output: bool = typer.Option(
        False,
        "--json",
        help="Output as JSON for scripting",
    ),
    detailed: bool = typer.Option(
        False,
        "--detailed", "-d",
        help="Show detailed info (architecture, modality, etc.)",
    ),
    all_providers: bool = typer.Option(
        False,
        "--all", "-a",
        help="Skip provider selection and show all models",
    ),
):
    """
    List available OpenRouter models for MAV and generation.

    \b
    Examples:
        cera models                    # Interactive: select provider first
        cera models --all              # Show all models (skip provider selection)
        cera models claude             # Search for Claude models
        cera models --provider openai  # List OpenAI models only
        cera models --free             # List only free models
        cera models --vision           # List models with image support
        cera models --detailed         # Show detailed info
        cera models --json             # Output as JSON for config files

    \b
    Legend:
        [F] = Free        [O] = Open Source/Weights
        [V] = Vision      [T] = Tool/Function calling
    """
    from rich.prompt import Prompt

    console.print(Panel.fit(
        "[bold blue]CERA[/bold blue] - Available OpenRouter Models",
        subtitle="Use model IDs in your config files",
    ))

    # Check for API key (uses .env first, then Convex fallback)
    from cera.config_loader import get_openrouter_api_key
    api_key = get_openrouter_api_key()
    if not api_key:
        console.print("\n[yellow]Warning:[/yellow] OpenRouter API key not found")
        console.print("[dim]Set OPENROUTER_API_KEY in .env or configure in web settings[/dim]\n")

    # Fetch models from OpenRouter
    console.print("[dim]Fetching models from OpenRouter...[/dim]")
    try:
        all_models = _fetch_openrouter_models()
    except Exception as e:
        console.print(f"[red]Error fetching models:[/red] {e}")
        raise typer.Exit(1)

    # Apply global filters first (these narrow down the set before provider selection)
    filtered_models = all_models
    if free_only:
        filtered_models = [m for m in filtered_models if m["is_free"]]
    if vision_only:
        filtered_models = [m for m in filtered_models if m["has_vision"]]
    if search:
        filtered_models = [m for m in filtered_models if search.lower() in m["id"].lower() or search.lower() in m["name"].lower()]

    # Track if a specific provider was selected (affects limit behavior)
    provider_selected = False

    # If provider specified via CLI, skip interactive selection
    if provider:
        filtered_models = [m for m in filtered_models if m["id"].startswith(f"{provider}/")]
        provider_selected = True
    # If search was specified, also skip interactive (show results directly)
    elif search or all_providers:
        pass  # Show all filtered results
    else:
        # Interactive provider selection
        # Group by provider and count
        provider_stats: dict[str, dict] = {}
        for model in filtered_models:
            p = model["provider"]
            if p not in provider_stats:
                provider_stats[p] = {"count": 0, "free": 0, "vision": 0, "oss": 0}
            provider_stats[p]["count"] += 1
            if model["is_free"]:
                provider_stats[p]["free"] += 1
            if model["has_vision"]:
                provider_stats[p]["vision"] += 1
            if model["is_oss"]:
                provider_stats[p]["oss"] += 1

        # Sort providers by model count (descending)
        sorted_providers = sorted(provider_stats.items(), key=lambda x: -x[1]["count"])

        # Display provider table
        console.print(f"\n[green]Found {len(filtered_models)} models from {len(provider_stats)} providers[/green]\n")

        provider_table = Table(show_header=True, header_style="bold", title="Select a Provider")
        provider_table.add_column("#", style="dim", width=4)
        provider_table.add_column("Provider", style="cyan")
        provider_table.add_column("Models", justify="right")
        provider_table.add_column("Free", justify="right", style="green")
        provider_table.add_column("Vision", justify="right", style="magenta")
        provider_table.add_column("OSS", justify="right", style="blue")

        provider_list = []
        for idx, (prov, stats) in enumerate(sorted_providers, 1):
            provider_list.append(prov)
            provider_table.add_row(
                str(idx),
                prov,
                str(stats["count"]),
                str(stats["free"]) if stats["free"] > 0 else "[dim]-[/dim]",
                str(stats["vision"]) if stats["vision"] > 0 else "[dim]-[/dim]",
                str(stats["oss"]) if stats["oss"] > 0 else "[dim]-[/dim]",
            )

        console.print(provider_table)
        console.print("\n[dim]Enter a number to select a provider, or 'all' to show all models[/dim]")
        console.print("[dim]You can also type the provider name directly (e.g., 'anthropic')[/dim]\n")

        # Get user selection
        selection = Prompt.ask("Select provider", default="1")

        provider_selected = False
        if selection.lower() == "all":
            pass  # Show all models
        elif selection.isdigit():
            idx = int(selection) - 1
            if 0 <= idx < len(provider_list):
                provider = provider_list[idx]
                filtered_models = [m for m in filtered_models if m["provider"] == provider]
                provider_selected = True
                console.print(f"\n[bold]Showing models from: {provider}[/bold]\n")
            else:
                console.print(f"[red]Invalid selection. Showing all models.[/red]")
        else:
            # Try to match by name
            matched = [p for p in provider_list if selection.lower() in p.lower()]
            if len(matched) == 1:
                provider = matched[0]
                filtered_models = [m for m in filtered_models if m["provider"] == provider]
                provider_selected = True
                console.print(f"\n[bold]Showing models from: {provider}[/bold]\n")
            elif len(matched) > 1:
                console.print(f"[yellow]Multiple matches: {', '.join(matched)}. Showing all.[/yellow]")
            else:
                console.print(f"[yellow]No provider matching '{selection}'. Showing all.[/yellow]")

    # Sort by provider, then by name
    filtered_models.sort(key=lambda x: (x["provider"], x["name"]))

    # Limit results (no limit when a specific provider is selected)
    total_count = len(filtered_models)
    if provider_selected:
        display_models = filtered_models  # Show all models for selected provider
    else:
        display_models = filtered_models[:limit]

    # JSON output
    if json_output:
        import json as json_module
        output_data = {
            "models": filtered_models,
            "total_count": total_count,
            "shown": len(display_models),
        }
        console.print(json_module.dumps(output_data, indent=2))
        return

    # Display results
    console.print(f"\n[green]Found {total_count} models[/green]", end="")
    if not provider_selected and total_count > limit:
        console.print(f" (showing first {limit}, use --provider or select a provider to see all)")
    else:
        console.print()

    _display_models_table(display_models, detailed)

    # Usage hints
    console.print("\n[bold]Usage in config.json:[/bold]")
    console.print("[dim]For MAV models (subject_profile.mav.models):[/dim]")
    console.print('  "mav": { "models": ["anthropic/claude-3.5-sonnet", "openai/gpt-4o", "google/gemini-2.5-flash"] }')
    console.print("\n[dim]For generation model (generation.model):[/dim]")
    console.print('  "generation": { "model": "anthropic/claude-sonnet-4" }')


@app.command()
def version():
    """Show CERA version information."""
    from cera import __version__

    console.print(f"CERA version [bold]{__version__}[/bold]")


if __name__ == "__main__":
    app()
