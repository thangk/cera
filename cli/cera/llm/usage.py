"""Usage Tracker - Accumulates token usage and computes costs from OpenRouter responses."""

import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import httpx


@dataclass
class LLMUsage:
    """A single LLM call's token usage."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str
    component: str = ""        # e.g. "sil.research", "aml", "heuristic", "rde"
    target: str = ""           # e.g. "100", "500" (dataset target size)
    run: str = ""              # e.g. "run1", "run2"
    timestamp: str = ""        # ISO 8601
    generation_id: str = ""    # OpenRouter generation ID for actual cost lookup


# Map component labels to pipeline phases
_COMPONENT_PHASE_MAP = {
    "sil.research": "composition",
    "sil.search": "composition",
    "sil.queries": "composition",
    "sil.answers": "composition",
    "sil.judge": "composition",
    "sil.classify": "composition",
    "sil.cluster": "composition",
    "sil.cluster_judge": "composition",
    "rde": "composition",
    "rde.subject": "composition",
    "rde.reviewer": "composition",
    "rde.query": "composition",
    "rde.domain": "composition",
    "rde.region": "composition",
    "aml": "generation",
    "aml.persona": "generation",
    "aml.patterns": "generation",
    "aml.structures": "generation",
    "heuristic": "generation",
    "context_extractor": "composition",
}


def _phase_for_component(component: str) -> str:
    """Map a component label to its pipeline phase."""
    if component in _COMPONENT_PHASE_MAP:
        return _COMPONENT_PHASE_MAP[component]
    # Fallback: check prefix
    for prefix in ("sil.", "rde.", "aml."):
        if component.startswith(prefix):
            return _COMPONENT_PHASE_MAP.get(prefix.rstrip("."), "unknown")
    return "unknown"


def _aggregate_records(records: list[LLMUsage], pricing: dict) -> dict:
    """Aggregate a list of records into a summary dict with cost."""
    prompt = sum(r.prompt_tokens for r in records)
    completion = sum(r.completion_tokens for r in records)
    total_cost = 0.0
    for r in records:
        p = pricing.get(r.model) or pricing.get(r.model.replace(":online", ""), {})
        total_cost += r.prompt_tokens * p.get("prompt", 0.0) + r.completion_tokens * p.get("completion", 0.0)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "total_calls": len(records),
        "estimated_cost_usd": total_cost,
    }


class UsageTracker:
    """
    Accumulates token usage across pipeline execution.

    Records usage from each OpenRouter API call and computes
    estimated costs by fetching model pricing.
    """

    def __init__(self):
        self._records: list[LLMUsage] = []
        self._model_pricing: dict[str, dict[str, float]] = {}
        # Current context for sequential labeling
        self._current_component: str = ""
        self._current_target: str = ""
        self._current_run: str = ""

    def set_context(self, component: str = "", target: str = "", run: str = ""):
        """Set tracking context. Subsequent records inherit these labels if their own fields are empty."""
        self._current_component = component
        self._current_target = target
        self._current_run = run

    def record(self, usage: LLMUsage):
        """Record a single LLM call's usage, applying current context for empty fields."""
        if not usage.component and self._current_component:
            usage.component = self._current_component
        if not usage.target and self._current_target:
            usage.target = self._current_target
        if not usage.run and self._current_run:
            usage.run = self._current_run
        if not usage.timestamp:
            usage.timestamp = datetime.now(timezone.utc).isoformat()
        self._records.append(usage)

    def merge(self, other: "UsageTracker"):
        """Merge records from another tracker into this one."""
        self._records.extend(other._records)
        # Merge pricing (other's pricing doesn't override ours)
        for model, pricing in other._model_pricing.items():
            if model not in self._model_pricing:
                self._model_pricing[model] = pricing

    def import_records(self, records: list[dict]):
        """Import serialized records (e.g. from RDE API response) into this tracker."""
        for r in records:
            self._records.append(LLMUsage(
                prompt_tokens=r.get("prompt_tokens", 0),
                completion_tokens=r.get("completion_tokens", 0),
                total_tokens=r.get("total_tokens", 0),
                model=r.get("model", ""),
                component=r.get("component", ""),
                target=r.get("target", ""),
                run=r.get("run", ""),
                timestamp=r.get("timestamp", ""),
                generation_id=r.get("generation_id", ""),
            ))

    def records_list(self) -> list[dict]:
        """Return all records as serializable dicts."""
        return [asdict(r) for r in self._records]

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

    def _build_tokens_report(self, job_id: str, method: str = "cera") -> dict:
        """Build the full tokens.json report structure."""
        pricing = self._model_pricing

        # Totals
        totals = _aggregate_records(self._records, pricing)

        # By phase → components
        by_phase: dict[str, dict] = {}
        for phase_name in ("composition", "generation", "evaluation"):
            phase_records = [r for r in self._records if _phase_for_component(r.component) == phase_name]
            phase_data = _aggregate_records(phase_records, pricing)

            # Group by component within phase
            components: dict[str, list[LLMUsage]] = {}
            for r in phase_records:
                components.setdefault(r.component, []).append(r)

            phase_data["components"] = {
                comp: _aggregate_records(recs, pricing)
                for comp, recs in sorted(components.items())
            }
            by_phase[phase_name] = phase_data

        # By model (with pricing info)
        by_model: dict[str, dict] = {}
        model_groups: dict[str, list[LLMUsage]] = {}
        for r in self._records:
            model_groups.setdefault(r.model, []).append(r)
        for model, recs in sorted(model_groups.items()):
            model_data = _aggregate_records(recs, pricing)
            # Include pricing rates
            p = pricing.get(model) or pricing.get(model.replace(":online", ""), {})
            model_data["pricing"] = {
                "prompt_per_token": p.get("prompt", 0.0),
                "completion_per_token": p.get("completion", 0.0),
            }
            by_model[model] = model_data

        # By target → runs
        by_target: dict[str, dict] = {}
        target_groups: dict[str, list[LLMUsage]] = {}
        for r in self._records:
            if r.target:
                target_groups.setdefault(r.target, []).append(r)
        for target, recs in sorted(target_groups.items(), key=lambda x: (x[0].isdigit(), int(x[0]) if x[0].isdigit() else 0)):
            target_data = _aggregate_records(recs, pricing)
            # Group by run within target
            run_groups: dict[str, list[LLMUsage]] = {}
            for r in recs:
                if r.run:
                    run_groups.setdefault(r.run, []).append(r)
            if run_groups:
                target_data["runs"] = {
                    run: _aggregate_records(run_recs, pricing)
                    for run, run_recs in sorted(run_groups.items())
                }
            by_target[target] = target_data

        return {
            "version": 1,
            "job_id": job_id,
            "method": method,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "totals": totals,
            "by_phase": by_phase,
            "by_model": by_model,
            "by_target": by_target,
            "records": self.records_list(),
        }

    def save_tokens_json(self, path: Path, job_id: str, method: str = "cera"):
        """Save the full tokens report to a JSON file. Can be called incrementally."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        report = self._build_tokens_report(job_id, method)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
