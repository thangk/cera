"""Usage Tracker - Accumulates token usage and computes costs from OpenRouter responses."""

from dataclasses import dataclass
import httpx


@dataclass
class LLMUsage:
    """A single LLM call's token usage."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str


class UsageTracker:
    """
    Accumulates token usage across pipeline execution.

    Records usage from each OpenRouter API call and computes
    estimated costs by fetching model pricing.
    """

    def __init__(self):
        self._records: list[LLMUsage] = []
        self._model_pricing: dict[str, dict[str, float]] = {}

    def record(self, usage: LLMUsage):
        """Record a single LLM call's usage."""
        self._records.append(usage)

    @property
    def total_prompt_tokens(self) -> int:
        """Total prompt tokens across all recorded calls."""
        return sum(r.prompt_tokens for r in self._records)

    @property
    def total_completion_tokens(self) -> int:
        """Total completion tokens across all recorded calls."""
        return sum(r.completion_tokens for r in self._records)

    @property
    def total_tokens(self) -> int:
        """Total tokens (prompt + completion) across all recorded calls."""
        return self.total_prompt_tokens + self.total_completion_tokens

    @property
    def call_count(self) -> int:
        """Number of LLM calls recorded."""
        return len(self._records)

    @property
    def models_used(self) -> list[str]:
        """Unique models used across all calls."""
        return list(set(r.model for r in self._records))

    async def fetch_pricing(self, api_key: str):
        """
        Fetch pricing for all models used from OpenRouter's models endpoint.

        Args:
            api_key: OpenRouter API key for authentication
        """
        models_needed = self.models_used
        if not models_needed:
            return

        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://openrouter.ai/api/v1/models",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()

        # Extract pricing for models we used
        for model_info in data.get("data", []):
            model_id = model_info.get("id", "")
            if model_id in models_needed or any(model_id.startswith(m.rstrip(":online")) for m in models_needed):
                pricing = model_info.get("pricing", {})
                prompt_price = pricing.get("prompt")
                completion_price = pricing.get("completion")
                if prompt_price is not None and completion_price is not None:
                    self._model_pricing[model_id] = {
                        "prompt": float(prompt_price),
                        "completion": float(completion_price),
                    }

    def compute_cost(self) -> dict:
        """
        Compute estimated cost based on recorded usage and fetched pricing.

        Returns:
            Dict with total_cost_usd, by_model breakdown, and total_tokens.
            Costs are 0.0 if pricing hasn't been fetched.
        """
        by_model: dict[str, dict] = {}
        total_cost = 0.0

        for record in self._records:
            model = record.model
            # Try exact match, then strip :online suffix
            pricing = self._model_pricing.get(model)
            if not pricing:
                base_model = model.replace(":online", "")
                pricing = self._model_pricing.get(base_model, {})

            prompt_cost = record.prompt_tokens * pricing.get("prompt", 0.0)
            completion_cost = record.completion_tokens * pricing.get("completion", 0.0)
            call_cost = prompt_cost + completion_cost

            if model not in by_model:
                by_model[model] = {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "cost_usd": 0.0,
                    "calls": 0,
                }

            by_model[model]["prompt_tokens"] += record.prompt_tokens
            by_model[model]["completion_tokens"] += record.completion_tokens
            by_model[model]["total_tokens"] += record.prompt_tokens + record.completion_tokens
            by_model[model]["cost_usd"] += call_cost
            by_model[model]["calls"] += 1
            total_cost += call_cost

        return {
            "total_cost_usd": total_cost,
            "total_tokens": self.total_tokens,
            "total_calls": self.call_count,
            "by_model": by_model,
        }

    def summary_dict(self) -> dict:
        """
        Return a serializable summary dict for .meta.json output.

        Returns:
            Dict with prompt_tokens, completion_tokens, total_tokens,
            estimated_cost_usd, and per-model breakdown.
        """
        cost_data = self.compute_cost()
        return {
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "total_calls": self.call_count,
            "estimated_cost_usd": cost_data["total_cost_usd"],
            "by_model": cost_data["by_model"],
        }
