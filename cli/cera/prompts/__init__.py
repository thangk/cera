"""Prompt Template Loader - Utilities for loading and formatting prompt templates."""

from pathlib import Path
from typing import Optional


PROMPTS_DIR = Path(__file__).parent


def load_prompt(category: str, name: str) -> str:
    """
    Load a prompt template from the prompts directory.

    Args:
        category: The category folder (e.g., 'aml', 'sil', 'mav')
        name: The prompt file name without extension (e.g., 'system', 'understand')

    Returns:
        The prompt template as a string

    Raises:
        FileNotFoundError: If the prompt file doesn't exist
    """
    path = PROMPTS_DIR / category / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"Prompt not found: {category}/{name}")
    return path.read_text(encoding="utf-8")


def format_prompt(template: str, **kwargs) -> str:
    """
    Format a prompt template with variables.

    Uses Python's str.format() for variable substitution.
    Variables in the template should be in {variable_name} format.

    Args:
        template: The prompt template string
        **kwargs: Variable values to substitute

    Returns:
        The formatted prompt string
    """
    return template.format(**kwargs)


def load_and_format(category: str, name: str, **kwargs) -> str:
    """
    Load a prompt template and format it with variables.

    Convenience function combining load_prompt and format_prompt.

    Args:
        category: The category folder (e.g., 'aml', 'sil', 'mav')
        name: The prompt file name without extension
        **kwargs: Variable values to substitute

    Returns:
        The formatted prompt string
    """
    template = load_prompt(category, name)
    return format_prompt(template, **kwargs)


def get_available_prompts() -> dict[str, list[str]]:
    """
    Get a dictionary of all available prompts organized by category.

    Returns:
        Dict mapping category names to lists of prompt names
    """
    prompts = {}
    for category_dir in PROMPTS_DIR.iterdir():
        if category_dir.is_dir() and not category_dir.name.startswith("_"):
            prompts[category_dir.name] = [
                f.stem for f in category_dir.glob("*.md")
            ]
    return prompts
