"""CERA FastAPI Server - HTTP API for web GUI integration."""

from fastapi import FastAPI, BackgroundTasks, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Union
import httpx
import asyncio
import json
import os
import re
from datetime import datetime
from pathlib import Path

app = FastAPI(
    title="CERA API",
    description="Context-Engineered Reviews Architecture - API Server",
    version="0.1.0",
)

# CORS middleware for web GUI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class MAVConfig(BaseModel):
    enabled: bool = False
    models: list[str] = []
    similarity_threshold: float = 0.75
    answer_threshold: float = 0.80
    max_queries: int = 30


class SubjectProfile(BaseModel):
    query: str
    region: str
    domain: Optional[str] = None  # Product/service domain (e.g., restaurant, laptop, hotel)
    category: Optional[str] = None  # Deprecated: use 'domain' instead
    sentiment_depth: str
    additional_context: Optional[str] = None  # Additional context about the subject
    mav: Optional[MAVConfig] = None
    aspect_categories: Optional[list[str]] = None

    @property
    def resolved_domain(self) -> str:
        """Get domain, falling back to deprecated category field."""
        return self.domain or self.category or "general"


class ReviewerProfile(BaseModel):
    age_range: Optional[list[int]] = None  # None when age ablation is disabled
    sex_distribution: dict[str, float]
    additional_context: Optional[str] = None
    persona_ratio: Optional[float] = 0.9  # 0.0-1.0, percentage of unique personas relative to review count


class NoiseConfig(BaseModel):
    typo_rate: float
    colloquialism: bool
    grammar_errors: bool
    preset: Optional[str] = None
    advanced: Optional[bool] = None
    use_ocr: Optional[bool] = None
    use_contextual: Optional[bool] = None


class PolarityConfig(BaseModel):
    positive: float
    neutral: float
    negative: float


class EdgeLengths(BaseModel):
    min_length: int = 1
    min_chance: float = 0.15  # 15% combined chance for minEdge range [min_length, normal_floor - 1]
    max_length: int = 15
    max_chance: float = 0.05  # 5% combined chance for maxEdge range [normal_ceiling + 1, max_length]


class AttributesProfile(BaseModel):
    polarity: PolarityConfig
    noise: NoiseConfig
    length_range: list[int]
    edge_lengths: Optional[EdgeLengths] = None
    temp_range: list[float] = [0.85, 0.95]  # LLM temperature range, optional with default
    cap_weights: Optional[dict] = None  # e.g., {"standard": 0.55, "lowercase": 0.20, "mixed": 0.15, "emphasis": 0.10}


class TargetDataset(BaseModel):
    """A single target dataset size with its own generation parameters."""
    count_mode: str = "sentences"  # "reviews" or "sentences"
    target_value: int = 100
    batch_size: int = 1
    request_size: int = 25
    total_runs: int = 1
    runs_mode: str = "parallel"  # "parallel" or "sequential"
    neb_depth: int = 0  # 0 = disabled


class GenerationConfig(BaseModel):
    count: int
    count_mode: str = "reviews"  # "reviews" or "sentences" - target mode
    target_sentences: Optional[int] = None  # Target sentence count when count_mode="sentences"
    batch_size: int
    request_size: int = 25
    provider: str
    model: str
    output_formats: Optional[list[str]] = None  # e.g., ["jsonl", "csv", "semeval_xml"]
    dataset_mode: str = "explicit"  # "explicit", "implicit", or "both"
    total_runs: int = 1  # Number of times to run generation (for research variability assessment)
    # NEB (Negative Example Buffer) - prevents generating similar reviews across batches
    neb_enabled: bool = True  # Enable NEB for diversity enforcement
    neb_depth: int = 0  # How many batches to remember (0=unlimited). Buffer size = neb_depth * request_size
    # Multi-model comparison
    models: Optional[list[str]] = None  # Multiple generation models (when set, overrides `model`)
    parallel_models: bool = True  # Run models concurrently (default ON)
    # Multi-target dataset support
    target_prefix: Optional[str] = None  # File naming prefix (e.g., "rq1-cera")
    targets: Optional[list[TargetDataset]] = None  # Multiple target dataset sizes
    parallel_targets: bool = True  # Run target datasets concurrently

    def get_effective_targets(self) -> list[TargetDataset]:
        """Return targets array, synthesizing from legacy fields if needed."""
        if self.targets:
            return self.targets
        return [TargetDataset(
            count_mode=self.count_mode,
            target_value=self.target_sentences if self.count_mode == "sentences" and self.target_sentences else self.count,
            batch_size=self.batch_size,
            request_size=self.request_size,
            total_runs=self.total_runs,
            runs_mode="sequential",
            neb_depth=self.neb_depth if self.neb_enabled else 0,
        )]


class AblationConfig(BaseModel):
    """Ablation study settings - toggle components on/off."""
    sil_enabled: bool = True
    mav_enabled: bool = True
    rgm_enabled: bool = True  # Controls PGM, WPM, SVM, and DEM (NEB, vocab tracking, opening directives, cap styles)
    polarity_enabled: bool = True
    noise_enabled: bool = True
    age_enabled: bool = True
    sex_enabled: bool = True


class JobConfig(BaseModel):
    subject_profile: SubjectProfile
    reviewer_profile: ReviewerProfile
    attributes_profile: AttributesProfile
    generation: GenerationConfig
    ablation: Optional[AblationConfig] = None


class JobRequest(BaseModel):
    jobId: str
    jobName: str = ""  # Job name for directory naming
    config: JobConfig
    apiKey: str
    jobsDirectory: str = "./jobs"  # Base directory for job files
    convexUrl: Optional[str] = None
    convexToken: Optional[str] = None


class CompositionRequest(BaseModel):
    """Request for composition-only execution (creates job dirs + runs SIL/RGM/ACM)."""
    jobId: str
    jobName: str
    config: JobConfig
    apiKey: str
    tavilyApiKey: Optional[str] = None
    jobsDirectory: str = "./jobs"
    convexUrl: Optional[str] = None
    convexToken: Optional[str] = None


class CompositionRequestV2(BaseModel):
    """
    Simplified request for composition - only requires job ID and Convex credentials.
    Job data and settings will be fetched directly from Convex.
    """
    jobId: str
    convexUrl: str
    convexToken: str


def sanitize_job_name(name: str) -> str:
    """Convert job name to safe directory name."""
    import re
    sanitized = name.lower()
    sanitized = re.sub(r'[^a-z0-9]+', '-', sanitized)  # Replace non-alphanumeric with hyphen
    sanitized = re.sub(r'^-|-$', '', sanitized)  # Trim leading/trailing hyphens
    return sanitized[:30] if sanitized else "job"  # Limit length


def create_job_directory(jobs_dir: str, job_id: str, job_name: str) -> dict[str, str]:
    """
    Create the job directory structure with all subdirectories.

    Structure:
    {jobs_dir}/{job_id}-{sanitized_name}/
    ├── contexts/      # Composition context files (shared across targets)
    ├── mavs/          # MAV raw data (shared)
    ├── reports/       # Analysis reports (shared)
    ├── datasets/      # Per-target subdirs via create_target_directory()
    └── config.json

    Returns:
        Dictionary with paths to each subdirectory
    """
    from pathlib import Path

    sanitized = sanitize_job_name(job_name)
    job_dir_name = f"{job_id}-{sanitized}" if sanitized else job_id
    job_dir = Path(jobs_dir) / job_dir_name

    # Create all subdirectories
    # Note: amls/ is NOT created here — created on demand by execute_generation
    # or by create_target_directory for multi-target jobs.
    subdirs = ["contexts", "mavs", "reports", "datasets"]
    paths = {"root": str(job_dir)}

    for subdir in subdirs:
        subdir_path = job_dir / subdir
        subdir_path.mkdir(parents=True, exist_ok=True)
        paths[subdir] = str(subdir_path)

    # Backward compat: paths["dataset"] aliases to paths["datasets"]
    paths["dataset"] = paths["datasets"]

    return paths


def create_target_directory(
    datasets_dir: str,
    target_value: int,
    count_mode: str,
    num_runs: int,
    models: list[str],
) -> dict[str, str]:
    """
    Create per-target directory structure under datasets/.

    Structure:
    datasets/{target_value}/
    ├── run1/
    │   ├── amls/          # AML prompts for this run
    │   ├── {model_slug}/  # Output per model
    │   └── {model_slug}/
    ├── run2/
    │   └── ...
    ├── metrics/           # MDQA results for this target size
    └── reviewer-personas/

    Returns:
        Dictionary with paths: target, metrics, reviewer-personas, and per-run/model paths
    """
    from pathlib import Path

    target_dir = Path(datasets_dir) / str(target_value)
    target_dir.mkdir(parents=True, exist_ok=True)

    metrics_dir = target_dir / "metrics"
    metrics_dir.mkdir(exist_ok=True)

    personas_dir = target_dir / "reviewer-personas"
    personas_dir.mkdir(exist_ok=True)

    paths = {
        "target": str(target_dir),
        "metrics": str(metrics_dir),
        "reviewer-personas": str(personas_dir),
        "runs": {},
    }

    for run_num in range(1, num_runs + 1):
        run_dir = target_dir / f"run{run_num}"
        amls_dir = run_dir / "amls"
        amls_dir.mkdir(parents=True, exist_ok=True)

        run_paths = {"root": str(run_dir), "amls": str(amls_dir), "models": {}}

        for model in models:
            slug = model.split("/")[-1] if "/" in model else model
            model_dir = run_dir / slug
            model_dir.mkdir(exist_ok=True)
            run_paths["models"][model] = str(model_dir)

        paths["runs"][run_num] = run_paths

    return paths


def _serialize_edge_lengths(edge_lengths) -> Optional[dict]:
    """Convert EdgeLengths (Pydantic model or dict) to a plain dict for JSON serialization."""
    if edge_lengths is None:
        return None
    if hasattr(edge_lengths, 'model_dump'):
        return edge_lengths.model_dump()
    if isinstance(edge_lengths, dict):
        return edge_lengths
    return None


def save_job_config(
    job_paths: dict,
    job_name: str,
    config: "JobConfig",
    phases: list[str],
    evaluation_config: Optional[dict] = None,
    reused_from: Optional[str] = None,
    reference_dataset: Optional[dict] = None,
) -> str:
    """
    Save job configuration to config.json in the job directory.

    This creates a portable configuration file that can be:
    1. Used to reproduce the job settings
    2. Referenced when reusing composition data in other jobs
    3. Used by CLI for standalone execution

    Returns the path to the saved config.json file.
    """
    import json
    from pathlib import Path
    from datetime import datetime

    config_data = {
        "version": "1.0",
        "name": job_name,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "phases": phases,
        "reused_from": reused_from,
    }

    # Add subject_profile if available
    if config and config.subject_profile:
        config_data["subject_profile"] = {
            "query": config.subject_profile.query,
            "region": config.subject_profile.region,
            "domain": getattr(config.subject_profile, 'resolved_domain', None) or getattr(config.subject_profile, 'domain', None),
            "additional_context": getattr(config.subject_profile, 'additional_context', ''),
            "sentiment_depth": config.subject_profile.sentiment_depth,
            "aspect_categories": getattr(config.subject_profile, 'aspect_categories', []) or [],
            "aspect_category_mode": getattr(config.subject_profile, 'aspect_category_mode', 'infer'),
        }
        if config.subject_profile.mav:
            config_data["subject_profile"]["mav"] = {
                "enabled": config.subject_profile.mav.enabled,
                "models": config.subject_profile.mav.models,
                "similarity_threshold": config.subject_profile.mav.similarity_threshold,
                "max_queries": getattr(config.subject_profile.mav, 'max_queries', 30),
            }

    # Add reviewer_profile if available
    if config and config.reviewer_profile:
        config_data["reviewer_profile"] = {
            "age_range": config.reviewer_profile.age_range,
            "sex_distribution": {
                "male": config.reviewer_profile.sex_distribution.male,
                "female": config.reviewer_profile.sex_distribution.female,
                "unspecified": config.reviewer_profile.sex_distribution.unspecified,
            } if hasattr(config.reviewer_profile.sex_distribution, 'male') else config.reviewer_profile.sex_distribution,
            "additional_context": getattr(config.reviewer_profile, 'additional_context', ''),
            "persona_ratio": getattr(config.reviewer_profile, 'persona_ratio', 0.9),
        }

    # Add attributes_profile if available
    if config and config.attributes_profile:
        config_data["attributes_profile"] = {
            "polarity": {
                "positive": config.attributes_profile.polarity.positive,
                "neutral": config.attributes_profile.polarity.neutral,
                "negative": config.attributes_profile.polarity.negative,
            },
            "noise": {
                "typo_rate": config.attributes_profile.noise.typo_rate,
                "colloquialism": config.attributes_profile.noise.colloquialism,
                "grammar_errors": config.attributes_profile.noise.grammar_errors,
                "preset": getattr(config.attributes_profile.noise, 'preset', 'moderate'),
            },
            "length_range": config.attributes_profile.length_range,
            "temperature_range": getattr(config.attributes_profile, 'temp_range', None) or getattr(config.attributes_profile, 'temperature_range', [0.85, 0.95]),
            "cap_weights": getattr(config.attributes_profile, 'cap_weights', None),
            "edge_lengths": _serialize_edge_lengths(getattr(config.attributes_profile, 'edge_lengths', None)),
        }

    # Add generation config if available
    if config and config.generation:
        gen_data = {
            "count": config.generation.count,
            "batch_size": config.generation.batch_size,
            "request_size": getattr(config.generation, 'request_size', 25),
            "model": config.generation.model,
            "provider": config.generation.provider,
            "output_formats": getattr(config.generation, 'output_formats', ["semeval_xml"]),
            "dataset_mode": getattr(config.generation, 'dataset_mode', 'both'),
            "total_runs": getattr(config.generation, 'total_runs', 1),
            # NEB (Negative Example Buffer) settings
            "neb_enabled": getattr(config.generation, 'neb_enabled', True),
            "neb_depth": getattr(config.generation, 'neb_depth', 0),
        }
        # Multi-target dataset support
        targets = getattr(config.generation, 'targets', None)
        if targets:
            gen_data["targets"] = [
                {
                    "count_mode": t.count_mode,
                    "target_value": t.target_value,
                    "batch_size": t.batch_size,
                    "request_size": t.request_size,
                    "total_runs": t.total_runs,
                    "runs_mode": t.runs_mode,
                    "neb_depth": getattr(t, 'neb_depth', 0),
                }
                for t in targets
            ]
        target_prefix = getattr(config.generation, 'target_prefix', None)
        if target_prefix:
            gen_data["target_prefix"] = target_prefix
        parallel_targets = getattr(config.generation, 'parallel_targets', None)
        if parallel_targets is not None:
            gen_data["parallel_targets"] = parallel_targets
        # Multi-model support
        models = getattr(config.generation, 'models', None)
        if models:
            gen_data["models"] = [m for m in models if m]
        parallel_models = getattr(config.generation, 'parallel_models', None)
        if parallel_models is not None:
            gen_data["parallel_models"] = parallel_models
        # Count mode
        count_mode = getattr(config.generation, 'count_mode', None)
        if count_mode:
            gen_data["count_mode"] = count_mode
        target_sentences = getattr(config.generation, 'target_sentences', None)
        if target_sentences:
            gen_data["target_sentences"] = target_sentences
        config_data["generation"] = gen_data

    # Add ablation settings if provided
    if config and config.ablation:
        config_data["ablation"] = {
            "sil_enabled": config.ablation.sil_enabled,
            "mav_enabled": config.ablation.mav_enabled,
            "rgm_enabled": config.ablation.rgm_enabled,
            "polarity_enabled": config.ablation.polarity_enabled,
            "noise_enabled": config.ablation.noise_enabled,
            "age_enabled": config.ablation.age_enabled,
            "sex_enabled": config.ablation.sex_enabled,
        }

    # Add evaluation config if provided
    if evaluation_config:
        config_data["evaluation"] = evaluation_config

    # Add reference dataset config if provided
    if reference_dataset:
        config_data["reference_dataset"] = reference_dataset

    config_path = Path(job_paths["root"]) / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2, default=str)

    return str(config_path)


def _parse_local_model(model_id: str) -> tuple[bool, str]:
    """Parse a model ID and detect if it's a local vLLM model.

    Local models are prefixed with 'local/' (e.g., 'local/Qwen/Qwen3-30B-A3B').
    Returns (is_local, actual_model_id) with the prefix stripped for local models.
    """
    if model_id.startswith("local/"):
        return True, model_id[len("local/"):]
    return False, model_id


async def _get_local_llm_settings(convex) -> tuple[str, str]:
    """Fetch local LLM endpoint and API key from Convex settings.

    Returns (endpoint, api_key). Raises ValueError if not configured.
    """
    settings = await convex.get_settings()
    if not settings:
        raise ValueError("Could not fetch settings from Convex")
    endpoint = settings.get("localLlmEndpoint", "")
    api_key = settings.get("localLlmApiKey", "")
    if not endpoint:
        raise ValueError("Local LLM endpoint not configured in Settings. Go to Settings > Local LLM Server.")
    # Ensure endpoint ends with /v1
    endpoint = endpoint.rstrip("/")
    if not endpoint.endswith("/v1"):
        endpoint += "/v1"
    return endpoint, api_key


async def check_openrouter_tier(api_key: str) -> dict:
    """
    Check OpenRouter account tier to determine rate limiting.

    Returns:
        dict with keys:
        - is_free_tier: bool - True if free tier (needs rate limiting)
        - rate_limit_requests: int - Requests per interval (e.g., 20)
        - rate_limit_interval: str - Interval string (e.g., "60s")
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://openrouter.ai/api/v1/auth/key",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=10.0,
            )
            if response.status_code == 200:
                data = response.json().get("data", {})
                rate_limit = data.get("rate_limit", {})
                return {
                    "is_free_tier": data.get("is_free_tier", True),
                    "rate_limit_requests": rate_limit.get("requests", 20),
                    "rate_limit_interval": rate_limit.get("interval", "60s"),
                }
    except Exception as e:
        print(f"Warning: Could not check OpenRouter tier: {e}")

    # Default to free tier behavior if check fails (safe fallback)
    return {
        "is_free_tier": True,
        "rate_limit_requests": 20,
        "rate_limit_interval": "60s",
    }


class PocketBaseClient:
    """Client for fast real-time progress updates via PocketBase SSE."""

    def __init__(self, url: str):
        self.url = url.rstrip("/")
        self.client = httpx.AsyncClient(timeout=5.0)
        self._records: dict[str, str] = {}   # job_id -> PB record ID
        self._state: dict[str, dict] = {}    # job_id -> current progress state

    async def update(self, job_id: str, **fields):
        """Update progress fields. Merges into local state and PATCHes PocketBase."""
        state = self._state.setdefault(job_id, {})
        state.update(fields)
        state["job_id"] = job_id

        record_id = self._records.get(job_id)
        try:
            if record_id:
                resp = await self.client.patch(
                    f"{self.url}/api/collections/job_progress/records/{record_id}",
                    json=state,
                )
                if resp.status_code == 404:
                    # Record was deleted (e.g. cleanup), recreate
                    self._records.pop(job_id, None)
                    record_id = None
            if not record_id:
                # Try to find existing record
                resp = await self.client.get(
                    f"{self.url}/api/collections/job_progress/records",
                    params={"filter": f'job_id="{job_id}"', "perPage": 1},
                )
                if resp.status_code == 200:
                    items = resp.json().get("items", [])
                    if items:
                        record_id = items[0]["id"]
                        self._records[job_id] = record_id
                        await self.client.patch(
                            f"{self.url}/api/collections/job_progress/records/{record_id}",
                            json=state,
                        )
                    else:
                        resp = await self.client.post(
                            f"{self.url}/api/collections/job_progress/records",
                            json=state,
                        )
                        if resp.status_code == 200:
                            self._records[job_id] = resp.json()["id"]
        except Exception as e:
            print(f"[PocketBase] Warning: progress update failed: {e}")

    async def add_log(self, job_id: str, level: str, phase: str, message: str):
        """Append a log entry to PocketBase (fast SQLite INSERT + SSE push)."""
        try:
            await self.client.post(
                f"{self.url}/api/collections/job_logs/records",
                json={"job_id": job_id, "level": level, "phase": phase, "message": message},
            )
        except Exception as e:
            print(f"[PocketBase] Warning: log write failed: {e}")

    async def cleanup(self, job_id: str):
        """Delete progress record for a completed job."""
        record_id = self._records.pop(job_id, None)
        self._state.pop(job_id, None)
        if record_id:
            try:
                await self.client.delete(
                    f"{self.url}/api/collections/job_progress/records/{record_id}",
                )
            except Exception:
                pass

    def get_state(self, job_id: str) -> dict:
        """Get current local progress state for final sync."""
        return self._state.get(job_id, {})

    async def fetch_state(self, job_id: str) -> dict:
        """Fetch actual progress state from PocketBase DB (not local cache).
        Used by complete_job to get the real values written by all client instances."""
        try:
            resp = await self.client.get(
                f"{self.url}/api/collections/job_progress/records",
                params={"filter": f'job_id="{job_id}"', "perPage": 1},
            )
            if resp.status_code == 200:
                items = resp.json().get("items", [])
                if items:
                    return items[0]
        except Exception:
            pass
        return self._state.get(job_id, {})


class ConvexClient:
    """Client for updating Convex database from Python."""

    # Progress-related mutations routed to PocketBase when available
    _PB_ROUTED_MUTATIONS = {
        "jobs:updateHeuristicProgress",
        "jobs:updateRunProgress",
        "jobs:updateTargetProgress",
    }

    def __init__(self, url: str, token: str, pocketbase_url: str = None):
        self.url = url
        self.token = token
        self.client = httpx.AsyncClient()
        self.pb = PocketBaseClient(pocketbase_url) if pocketbase_url else None

    @property
    def has_fast_updates(self) -> bool:
        """True when PocketBase is available for high-frequency progress updates."""
        return self.pb is not None

    async def get_job(self, job_id: str) -> Optional[dict]:
        """Fetch job data from Convex (public query, no auth needed)."""
        try:
            response = await self.client.post(
                f"{self.url}/api/query",
                headers={"Content-Type": "application/json"},
                json={
                    "path": "jobs:get",
                    "args": {"id": job_id},
                },
            )
            response.raise_for_status()
            result = response.json()
            if result.get("status") == "success":
                return result.get("value")
            return None
        except Exception as e:
            print(f"Warning: Failed to get job from Convex: {e}")
            return None

    async def get_settings(self) -> Optional[dict]:
        """Fetch settings from Convex (public query, no auth needed)."""
        try:
            response = await self.client.post(
                f"{self.url}/api/query",
                headers={"Content-Type": "application/json"},
                json={
                    "path": "settings:get",
                    "args": {},
                },
            )
            response.raise_for_status()
            result = response.json()
            if result.get("status") == "success":
                return result.get("value")
            return None
        except Exception as e:
            print(f"Warning: Failed to get settings from Convex: {e}")
            return None

    async def start_composing(self, job_id: str, job_dir: str):
        """Mark job as composing and store job directory."""
        try:
            response = await self.client.post(
                f"{self.url}/api/mutation",
                headers={
                    "Authorization": f"Convex {self.token}",
                    "Content-Type": "application/json",
                },
                json={
                    "path": "jobs:startComposing",
                    "args": {"jobId": job_id, "jobDir": job_dir},
                },
            )
            response.raise_for_status()
        except Exception as e:
            # Check if error is because job is already composing (expected case)
            error_msg = str(e)
            if "already" in error_msg.lower() or "composing" in error_msg.lower() or "pending" in error_msg.lower():
                print(f"[Composition] Job already in composing state - continuing")
            else:
                print(f"Warning: Failed to start composing in Convex: {e}")

    async def update_composition_progress(self, job_id: str, progress: int, phase: str):
        """Update composition progress. Routes to PocketBase when available."""
        if self.pb:
            await self.pb.update(job_id, progress=progress, current_phase=phase)
            return
        try:
            response = await self.client.post(
                f"{self.url}/api/mutation",
                headers={
                    "Authorization": f"Convex {self.token}",
                    "Content-Type": "application/json",
                },
                json={
                    "path": "jobs:updateCompositionProgress",
                    "args": {
                        "jobId": job_id,
                        "progress": progress,
                        "currentPhase": phase,
                    },
                },
            )
            response.raise_for_status()
            result = response.json()
            if result.get("status") == "error":
                print(f"Warning: Convex mutation error: {result.get('errorMessage', 'Unknown error')}")
        except Exception as e:
            print(f"Warning: Failed to update composition progress in Convex: {e}")

    async def complete_composition(self, job_id: str):
        """Mark composition as complete in Convex."""
        try:
            response = await self.client.post(
                f"{self.url}/api/mutation",
                headers={
                    "Authorization": f"Convex {self.token}",
                    "Content-Type": "application/json",
                },
                json={
                    "path": "jobs:completeComposition",
                    "args": {"jobId": job_id},
                },
            )
            response.raise_for_status()
        except Exception as e:
            print(f"Warning: Failed to complete composition in Convex: {e}")

    async def update_progress(self, job_id: str, progress: int, phase: str):
        """Update job progress. Routes to PocketBase when available."""
        if self.pb:
            await self.pb.update(job_id, progress=progress, current_phase=phase)
            return
        try:
            response = await self.client.post(
                f"{self.url}/api/mutation",
                headers={
                    "Authorization": f"Convex {self.token}",
                    "Content-Type": "application/json",
                },
                json={
                    "path": "jobs:updateProgress",
                    "args": {
                        "jobId": job_id,
                        "progress": progress,
                        "currentPhase": phase,
                    },
                },
            )
            response.raise_for_status()
        except Exception as e:
            print(f"Warning: Failed to update Convex progress: {e}")

    async def add_log(self, job_id: str, level: str, phase: str, message: str):
        """Add a log entry. Routes to PocketBase when available."""
        if self.pb:
            await self.pb.add_log(job_id, level, phase, message)
            return
        try:
            response = await self.client.post(
                f"{self.url}/api/mutation",
                headers={
                    "Authorization": f"Convex {self.token}",
                    "Content-Type": "application/json",
                },
                json={
                    "path": "logs:add",
                    "args": {
                        "jobId": job_id,
                        "level": level,
                        "phase": phase,
                        "message": message,
                    },
                },
            )
            response.raise_for_status()
        except Exception as e:
            print(f"Warning: Failed to add Convex log: {e}")

    async def complete_job(self, job_id: str):
        """Mark job as completed in Convex."""
        try:
            response = await self.client.post(
                f"{self.url}/api/mutation",
                headers={
                    "Authorization": f"Convex {self.token}",
                    "Content-Type": "application/json",
                },
                json={
                    "path": "jobs:complete",
                    "args": {"jobId": job_id},
                },
            )
            response.raise_for_status()
        except Exception as e:
            print(f"Warning: Failed to complete Convex job: {e}")
        # Sync final progress snapshot to Convex and clean up PocketBase
        if self.pb:
            # fetch_state reads from PB's REST API (not local cache) to get
            # values written by ALL ConvexClient instances (generation spawns
            # separate clients per target/run, each with its own local _state)
            state = await self.pb.fetch_state(job_id)
            headers = {"Authorization": f"Convex {self.token}", "Content-Type": "application/json"}
            # Sync counts
            for field, path in [
                ("generated_count", "jobs:updateGeneratedCount"),
                ("generated_sentences", "jobs:updateGeneratedSentences"),
            ]:
                if state.get(field):
                    try:
                        await self.client.post(
                            f"{self.url}/api/mutation", headers=headers,
                            json={"path": path, "args": {"jobId": job_id, "count": state[field]}},
                        )
                    except Exception:
                        pass
            # Sync currentRun/totalRuns so completed jobs show correct run counter
            if state.get("current_run") and state.get("total_runs"):
                try:
                    await self.client.post(
                        f"{self.url}/api/mutation", headers=headers,
                        json={"path": "jobs:updateCurrentRun", "args": {
                            "jobId": job_id,
                            "currentRun": state["current_run"],
                            "totalRuns": state["total_runs"],
                        }},
                    )
                except Exception:
                    pass
            # Sync target_progress so completed jobs show correct target statuses
            if state.get("target_progress"):
                for tp in state["target_progress"]:
                    try:
                        await self.client.post(
                            f"{self.url}/api/mutation", headers=headers,
                            json={"path": "jobs:updateTargetProgress", "args": {**tp, "jobId": job_id}},
                        )
                    except Exception:
                        pass
            await self.pb.cleanup(job_id)

    async def fail_job(self, job_id: str, error: str):
        """Mark job as failed in Convex."""
        try:
            response = await self.client.post(
                f"{self.url}/api/mutation",
                headers={
                    "Authorization": f"Convex {self.token}",
                    "Content-Type": "application/json",
                },
                json={
                    "path": "jobs:setFailed",
                    "args": {"jobId": job_id, "error": error},
                },
            )
            response.raise_for_status()
        except Exception as e:
            print(f"Warning: Failed to fail Convex job: {e}")
        # Clean up PocketBase on failure too
        if self.pb:
            await self.pb.cleanup(job_id)

    async def run_mutation(self, path: str, args: dict):
        """Run an arbitrary Convex mutation. Routes progress mutations to PocketBase."""
        if self.pb and path in self._PB_ROUTED_MUTATIONS:
            await self._route_progress_mutation(path, args)
            return None
        try:
            response = await self.client.post(
                f"{self.url}/api/mutation",
                headers={
                    "Authorization": f"Convex {self.token}",
                    "Content-Type": "application/json",
                },
                json={"path": path, "args": args},
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Warning: Failed to run Convex mutation {path}: {e}")
            return None

    async def _route_progress_mutation(self, path: str, args: dict):
        """Route progress-related mutations to PocketBase."""
        job_id = args.get("jobId", "")
        fields = {k: v for k, v in args.items() if k != "jobId"}

        if path == "jobs:updateHeuristicProgress":
            await self.pb.update(job_id, heuristic_progress=fields)

        elif path == "jobs:updateRunProgress":
            state = self.pb._state.setdefault(job_id, {})
            runs = state.get("run_progress") or []
            run_num = args.get("run")
            found = False
            for r in runs:
                if r.get("run") == run_num:
                    r.update(fields)
                    found = True
                    break
            if not found:
                runs.append(fields)
            await self.pb.update(job_id, run_progress=runs)

        elif path == "jobs:updateTargetProgress":
            state = self.pb._state.setdefault(job_id, {})
            targets = state.get("target_progress") or []
            idx = args.get("targetIndex")
            found = False
            for t in targets:
                if t.get("targetIndex") == idx:
                    t.update(fields)
                    found = True
                    break
            if not found:
                targets.append(fields)
            await self.pb.update(job_id, target_progress=targets)

    async def update_generated_count(self, job_id: str, count: int):
        """Update the generated review count. Routes to PocketBase when available."""
        if self.pb:
            await self.pb.update(job_id, generated_count=count)
            return
        try:
            response = await self.client.post(
                f"{self.url}/api/mutation",
                headers={
                    "Authorization": f"Convex {self.token}",
                    "Content-Type": "application/json",
                },
                json={
                    "path": "jobs:updateGeneratedCount",
                    "args": {"jobId": job_id, "count": count},
                },
            )
            response.raise_for_status()
        except Exception as e:
            print(f"Warning: Failed to update Convex generated count: {e}")

    async def update_failed_count(self, job_id: str, count: int):
        """Update the failed review count. Routes to PocketBase when available."""
        if self.pb:
            await self.pb.update(job_id, failed_count=count)
            return
        try:
            response = await self.client.post(
                f"{self.url}/api/mutation",
                headers={
                    "Authorization": f"Convex {self.token}",
                    "Content-Type": "application/json",
                },
                json={
                    "path": "jobs:updateFailedCount",
                    "args": {"jobId": job_id, "count": count},
                },
            )
            response.raise_for_status()
        except Exception as e:
            print(f"Warning: Failed to update Convex failed count: {e}")

    async def update_generated_sentences(self, job_id: str, count: int):
        """Update the generated sentence count. Routes to PocketBase when available."""
        if self.pb:
            await self.pb.update(job_id, generated_sentences=count)
            return
        try:
            response = await self.client.post(
                f"{self.url}/api/mutation",
                headers={
                    "Authorization": f"Convex {self.token}",
                    "Content-Type": "application/json",
                },
                json={
                    "path": "jobs:updateGeneratedSentences",
                    "args": {"jobId": job_id, "count": count},
                },
            )
            response.raise_for_status()
        except Exception as e:
            print(f"Warning: Failed to update Convex generated sentences: {e}")

    async def set_evaluation_device(self, job_id: str, device_type: str, device_name: str = None):
        """Set the evaluation device (GPU/CPU) in Convex."""
        try:
            device = {"type": device_type}
            if device_name:
                device["name"] = device_name
            response = await self.client.post(
                f"{self.url}/api/mutation",
                headers={
                    "Authorization": f"Convex {self.token}",
                    "Content-Type": "application/json",
                },
                json={
                    "path": "jobs:setEvaluationDevice",
                    "args": {"jobId": job_id, "device": device},
                },
            )
            response.raise_for_status()
        except Exception as e:
            print(f"Warning: Failed to set evaluation device in Convex: {e}")

    async def create_dataset(
        self,
        job_id: str,
        name: str,
        subject: str,
        category: str,
        review_count: int,
        metrics: dict,
        output_path: str,
    ):
        """Create a dataset entry in Convex."""
        try:
            response = await self.client.post(
                f"{self.url}/api/mutation",
                headers={
                    "Authorization": f"Convex {self.token}",
                    "Content-Type": "application/json",
                },
                json={
                    "path": "datasets:create",
                    "args": {
                        "jobId": job_id,
                        "name": name,
                        "subject": subject,
                        "category": category,
                        "reviewCount": review_count,
                        "metrics": metrics,
                        "outputPath": output_path,
                    },
                },
            )
            response.raise_for_status()
        except Exception as e:
            print(f"Warning: Failed to create Convex dataset: {e}")

    async def update_current_run(self, job_id: str, current_run: int, total_runs: int):
        """Update the current run number in Convex (for multi-run jobs)."""
        try:
            response = await self.client.post(
                f"{self.url}/api/mutation",
                headers={
                    "Authorization": f"Convex {self.token}",
                    "Content-Type": "application/json",
                },
                json={
                    "path": "jobs:updateCurrentRun",
                    "args": {
                        "jobId": job_id,
                        "currentRun": current_run,
                        "totalRuns": total_runs,
                    },
                },
            )
            response.raise_for_status()
        except Exception as e:
            print(f"Warning: Failed to update current run in Convex: {e}")

    async def update_actual_cost(self, job_id: str, actual_cost: dict):
        """Update actual cost from usage tracker after job completion."""
        try:
            response = await self.client.post(
                f"{self.url}/api/mutation",
                headers={
                    "Authorization": f"Convex {self.token}",
                    "Content-Type": "application/json",
                },
                json={
                    "path": "jobs:updateActualCost",
                    "args": {"jobId": job_id, "actualCost": actual_cost},
                },
            )
            response.raise_for_status()
        except Exception as e:
            print(f"Warning: Failed to update actual cost in Convex: {e}")

    async def save_per_run_metrics(
        self, job_id: str, run: int, dataset_file: str, metrics: dict
    ):
        """Save evaluation metrics for a specific run (for multi-run jobs)."""
        try:
            response = await self.client.post(
                f"{self.url}/api/mutation",
                headers={
                    "Authorization": f"Convex {self.token}",
                    "Content-Type": "application/json",
                },
                json={
                    "path": "jobs:savePerRunMetrics",
                    "args": {
                        "jobId": job_id,
                        "run": run,
                        "datasetFile": dataset_file,
                        "metrics": metrics,
                    },
                },
            )
            response.raise_for_status()
        except Exception as e:
            print(f"Warning: Failed to save per-run metrics in Convex: {e}")

    async def save_average_metrics(self, job_id: str, average_metrics: dict):
        """Save average metrics across all runs (for multi-run jobs)."""
        try:
            response = await self.client.post(
                f"{self.url}/api/mutation",
                headers={
                    "Authorization": f"Convex {self.token}",
                    "Content-Type": "application/json",
                },
                json={
                    "path": "jobs:saveAverageMetrics",
                    "args": {"jobId": job_id, "averageMetrics": average_metrics},
                },
            )
            response.raise_for_status()
        except Exception as e:
            print(f"Warning: Failed to save average metrics in Convex: {e}")

    async def update_model_progress(
        self, job_id: str, model: str, model_slug: str,
        generated: int, failed: int, target: int, progress: int,
        status: str, eval_progress: int = 0,
    ):
        """Update per-model progress. Sends to both PocketBase (real-time) and Convex (persistent)."""
        if self.pb:
            entry = {
                "model": model, "modelSlug": model_slug,
                "generated": generated, "failed": failed, "target": target,
                "progress": progress, "status": status, "evalProgress": eval_progress,
            }
            state = self.pb._state.setdefault(job_id, {})
            models = state.get("model_progress") or []
            found = False
            for m in models:
                if m.get("model") == model:
                    m.update(entry)
                    found = True
                    break
            if not found:
                models.append(entry)
            await self.pb.update(job_id, model_progress=models)
            # Also persist to Convex (don't return early)
        try:
            response = await self.client.post(
                f"{self.url}/api/mutation",
                headers={
                    "Authorization": f"Convex {self.token}",
                    "Content-Type": "application/json",
                },
                json={
                    "path": "jobs:updateModelProgress",
                    "args": {
                        "jobId": job_id,
                        "model": model,
                        "modelSlug": model_slug,
                        "generated": generated,
                        "failed": failed,
                        "target": target,
                        "progress": progress,
                        "status": status,
                        "evalProgress": eval_progress,
                    },
                },
            )
            response.raise_for_status()
        except Exception as e:
            print(f"Warning: Failed to update model progress in Convex: {e}")

    async def save_per_model_metrics(
        self, job_id: str, model: str, model_slug: str,
        metrics: dict, conformity: dict = None, per_run_metrics: list = None,
    ):
        """Save per-model evaluation metrics for multi-model jobs."""
        try:
            args = {
                "jobId": job_id,
                "model": model,
                "modelSlug": model_slug,
                "metrics": metrics,
            }
            if conformity:
                args["conformity"] = conformity
            if per_run_metrics:
                args["perRunMetrics"] = per_run_metrics
            response = await self.client.post(
                f"{self.url}/api/mutation",
                headers={
                    "Authorization": f"Convex {self.token}",
                    "Content-Type": "application/json",
                },
                json={
                    "path": "jobs:savePerModelMetrics",
                    "args": args,
                },
            )
            response.raise_for_status()
        except Exception as e:
            print(f"Warning: Failed to save per-model metrics in Convex: {e}")


async def execute_composition(
    job_id: str,
    job_name: str,
    config: JobConfig,
    api_key: str,
    tavily_api_key: Optional[str],
    jobs_directory: str,
    convex_url: Optional[str],
    convex_token: Optional[str],
) -> dict:
    """
    Execute only the COMPOSITION phase of the CERA pipeline.

    This creates job directories and generates:
    - subject-context.json (from SIL with optional MAV)
    - reviewers-context.json (from RGM)
    - attributes-context.json (from ACM)

    Returns the job paths dictionary.
    """
    import json
    import math
    from pathlib import Path
    from datetime import datetime

    from cera.pipeline.composition.sil import (
        SubjectIntelligenceLayer,
        MAVConfig as SILMAVConfig,
    )

    convex = None
    if convex_url and convex_token:
        convex = ConvexClient(convex_url, convex_token, pocketbase_url=os.environ.get("POCKETBASE_URL"))

    # Create job directory structure
    job_paths = create_job_directory(jobs_directory, job_id, job_name)

    try:
        # Mark job as composing
        if convex:
            await convex.start_composing(job_id, job_paths["root"])
            await convex.add_log(job_id, "INFO", "SIL", "Starting composition phase...")
            await convex.add_log(job_id, "INFO", "SIL", f"Job directory: {job_paths['root']}")

        # ========================================
        # Phase 1: SIL - Subject Intelligence Layer (with MAV)
        # ========================================
        mav_enabled = config.subject_profile.mav.enabled if config.subject_profile.mav else False
        # Ablation override: force MAV off if ablation says so
        if config.ablation and not config.ablation.mav_enabled:
            mav_enabled = False
        mav_models = config.subject_profile.mav.models if config.subject_profile.mav else []
        mav_threshold = config.subject_profile.mav.similarity_threshold if config.subject_profile.mav else 0.85
        mav_answer_threshold = getattr(config.subject_profile.mav, 'answer_threshold', 0.80) if config.subject_profile.mav else 0.80
        mav_max_queries = getattr(config.subject_profile.mav, 'max_queries', 30) if config.subject_profile.mav else 30

        if convex:
            await convex.update_composition_progress(job_id, 0, "SIL")
            await convex.add_log(
                job_id,
                "INFO",
                "SIL",
                f"Gathering intelligence for: {config.subject_profile.query}",
            )
            if mav_enabled and len(mav_models) >= 2:
                await convex.add_log(
                    job_id,
                    "INFO",
                    "MAV",
                    f"MAV enabled with {len(mav_models)} models: {', '.join(mav_models)}",
                )

        # Configure MAV
        sil_mav_config = SILMAVConfig(
            enabled=mav_enabled and len(mav_models) >= 2,
            models=mav_models,
            similarity_threshold=mav_threshold,
            answer_threshold=mav_answer_threshold,
            max_queries=mav_max_queries,
        )

        # Initialize SIL with MAV config
        # Create async log callback that sends logs to Convex (with optional progress)
        async def sil_log_callback(level: str, phase: str, message: str, progress: int = None):
            if convex:
                await convex.add_log(job_id, level, phase, message)
                if progress is not None:
                    await convex.update_composition_progress(job_id, progress, phase)

        sil_enabled = config.ablation.sil_enabled if config.ablation else True
        sil = SubjectIntelligenceLayer(
            api_key=api_key,
            mav_config=sil_mav_config,
            tavily_api_key=tavily_api_key,
            log_callback=sil_log_callback if convex else None,
            sil_enabled=sil_enabled,
        )

        if convex:
            await convex.update_composition_progress(job_id, 5, "SIL")
            if not sil_enabled:
                await convex.add_log(job_id, "INFO", "SIL", "SIL disabled — using LLM parametric knowledge only")
            if mav_enabled:
                await convex.add_log(job_id, "INFO", "MAV", "Starting multi-agent verification...")

        # Gather intelligence (this runs MAV if enabled)
        mav_result = await sil.gather_intelligence(
            query=config.subject_profile.query,
            region=config.subject_profile.region,
            domain=config.subject_profile.resolved_domain,
            sentiment_depth=config.subject_profile.sentiment_depth,
            additional_context=getattr(config.subject_profile, 'additional_context', None),
            aspect_categories=getattr(config.subject_profile, 'aspect_categories', None),
        )

        if convex:
            await convex.update_composition_progress(job_id, 25, "SIL")
            if mav_result.context.mav_verified:
                report = mav_result.query_pool_report
                consensus_msg = f"{report.queries_with_consensus}/{report.queries_after_dedup} queries reached consensus" if report else f"{mav_result.facts_verified} facts verified"
                await convex.add_log(
                    job_id,
                    "INFO",
                    "MAV",
                    f"MAV complete: {consensus_msg}",
                )

        # Build subject context from MAV result
        ctx = mav_result.context
        subject_context = {
            "query": ctx.subject,
            "region": config.subject_profile.region,
            "domain": config.subject_profile.resolved_domain,
            "sentiment_depth": config.subject_profile.sentiment_depth,
            "characteristics": ctx.features,
            "positives": ctx.pros,
            "negatives": ctx.cons,
            "use_cases": ctx.use_cases,
            "availability": ctx.availability,
            "mav_verified": ctx.mav_verified,
            "search_sources": ctx.search_sources,
            "mav_stats": {
                "total_facts_extracted": mav_result.total_facts_extracted,
                "facts_verified": mav_result.facts_verified,
                "facts_rejected": mav_result.facts_rejected,
                "verification_rate": (
                    mav_result.facts_verified / mav_result.total_facts_extracted
                    if mav_result.total_facts_extracted > 0 else 0
                ),
            },
        }

        # Save subject context
        subject_context_path = Path(job_paths["contexts"]) / "subject-context.json"
        with open(subject_context_path, "w") as f:
            json.dump(subject_context, f, indent=2)

        # Save verified facts (entity-clustered) if available
        if mav_result.verified_facts:
            verified_facts_path = Path(job_paths["contexts"]) / "verified-facts.json"
            with open(verified_facts_path, "w") as f:
                json.dump(mav_result.verified_facts, f, indent=2)
            if convex:
                n_ent = mav_result.verified_facts.get("total_entities", 0)
                await convex.add_log(job_id, "INFO", "SIL", f"Saved verified-facts.json with {n_ent} entities")

        # Save MAV raw data to mavs/ directory
        mavs_dir = Path(job_paths["mavs"])
        timestamp = datetime.utcnow().isoformat() + "Z"

        for model_data in mav_result.model_data:
            if model_data.error:
                if convex:
                    await convex.add_log(
                        job_id,
                        "WARN",
                        "MAV",
                        f"Model {model_data.model} failed: {model_data.error}",
                    )
                continue

            # Create model-specific directory
            model_dir = mavs_dir / model_data.sanitized_model
            model_dir.mkdir(parents=True, exist_ok=True)

            # Save understanding.md
            if model_data.understanding:
                understanding_content = f"""# MAV Understanding - {model_data.model}

## Subject
{config.subject_profile.query}

## Subject Type
{model_data.understanding.subject_type}

## Relevant Aspects
{chr(10).join(f'- {aspect}' for aspect in model_data.understanding.relevant_aspects)}

## Timestamp
{timestamp}
"""
                (model_dir / "understanding.md").write_text(understanding_content, encoding="utf-8")

            # Save queries.json - Factual queries this model generated
            if model_data.queries_generated:
                queries_data = {
                    "model": model_data.model,
                    "subject": config.subject_profile.query,
                    "queries": model_data.queries_generated,
                    "count": len(model_data.queries_generated),
                    "timestamp": timestamp,
                }
                (model_dir / "queries.json").write_text(json.dumps(queries_data, indent=2), encoding="utf-8")

            # Save answers.json - This model's answers to pooled queries
            if model_data.answers:
                answers_data = {
                    "model": model_data.model,
                    "subject": config.subject_profile.query,
                    "answers": [
                        {"query_id": a.query_id, "response": a.response, "confidence": a.confidence}
                        for a in model_data.answers
                    ],
                    "count": len(model_data.answers),
                    "timestamp": timestamp,
                }
                (model_dir / "answers.json").write_text(json.dumps(answers_data, indent=2), encoding="utf-8")

        # ========================================
        # Generate MAV Report
        # ========================================
        reports_dir = Path(job_paths["reports"])

        if mav_result.query_pool_report:
            report = mav_result.query_pool_report

            # Build MAV report JSON (query-based format)
            mav_report = {
                "generated_at": timestamp,
                "subject": config.subject_profile.query,
                "additional_context": getattr(config.subject_profile, 'additional_context', None),
                "config": {
                    "models": [md.model for md in mav_result.model_data if not md.error],
                    "model_count": len([md for md in mav_result.model_data if not md.error]),
                    "answer_similarity_threshold": report.threshold_used,
                    "max_queries": mav_max_queries,
                    "consensus_method": "LLM-judged agreement voting",
                },
                "summary": {
                    "total_queries_generated": report.total_queries_generated,
                    "queries_after_dedup": report.queries_after_dedup,
                    "queries_with_consensus": report.queries_with_consensus,
                    "queries_without_consensus": report.queries_without_consensus,
                    "consensus_rate": (
                        report.queries_with_consensus / report.queries_after_dedup
                        if report.queries_after_dedup > 0 else 0
                    ),
                    "used_fallback": report.used_fallback,
                },
                "per_query_consensus": [
                    {
                        "query_id": r.query_id,
                        "query": r.query,
                        "consensus_reached": r.consensus_reached,
                        "consensus_answer": r.consensus_answer,
                        "answers": [
                            {"model": a.model, "response": a.response, "confidence": a.confidence}
                            for a in r.answers
                        ],
                        "agreeing_models": r.agreeing_models,
                        "pairwise_similarities": r.pairwise_similarities,
                        "agreement_count": r.agreement_count,
                        "agreement_votes": r.agreement_votes,
                        "total_points": r.total_points,
                        "points_by_source": r.points_by_source,
                    }
                    for r in report.per_query_results
                ],
                "classified_facts": {
                    "characteristics": mav_result.context.features,
                    "positives": mav_result.context.pros,
                    "negatives": mav_result.context.cons,
                    "use_cases": mav_result.context.use_cases,
                },
            }

            # Save mav-report.json to mavs/ dir (alongside per-model raw data)
            mav_report_path = mavs_dir / "mav-report.json"
            with open(mav_report_path, "w", encoding="utf-8") as f:
                json.dump(mav_report, f, indent=2)

            # Generate mav-summary.csv for paper tables
            import csv
            summary_path = mavs_dir / "mav-summary.csv"
            with open(summary_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Metric", "Value"])
                writer.writerow(["subject", config.subject_profile.query])
                writer.writerow(["models_used", ",".join(md.model for md in mav_result.model_data if not md.error)])
                writer.writerow(["total_queries_generated", report.total_queries_generated])
                writer.writerow(["queries_after_dedup", report.queries_after_dedup])
                writer.writerow(["queries_with_consensus", report.queries_with_consensus])
                writer.writerow(["consensus_rate", f"{mav_report['summary']['consensus_rate']:.4f}"])
                writer.writerow(["answer_threshold", report.threshold_used])
                writer.writerow(["consensus_method", "LLM-judged agreement voting"])
                writer.writerow(["used_fallback", str(report.used_fallback)])

            if convex:
                await convex.add_log(job_id, "INFO", "MAV", "Generated MAV reports: mav-report.json, mav-summary.csv")

        if convex:
            await convex.update_composition_progress(job_id, 35, "SIL")
            await convex.add_log(job_id, "INFO", "SIL", "Subject context saved")

        # ========================================
        # Phase 2: RGM - Reviewer Generation Module
        # ========================================
        # Check ablation settings for age and sex
        age_enabled = getattr(config.ablation, 'age_enabled', True) if config.ablation else True
        sex_enabled = getattr(config.ablation, 'sex_enabled', True) if config.ablation else True

        if convex:
            await convex.update_composition_progress(job_id, 50, "RGM")
            if age_enabled:
                age_info = f"age: {config.reviewer_profile.age_range}"
            else:
                age_info = "age: disabled (ablation)"
            await convex.add_log(
                job_id,
                "INFO",
                "RGM",
                f"Generating reviewer profiles ({age_info})",
            )

        # Reviewers context contains ONLY the specs/distribution
        # Respect ablation settings: None for age_range if disabled, 100% unspecified for sex if disabled
        reviewers_context = {
            "age_range": config.reviewer_profile.age_range if age_enabled else None,
            "sex_distribution": config.reviewer_profile.sex_distribution if sex_enabled else {"male": 0.0, "female": 0.0, "unspecified": 1.0},
            "additional_context": config.reviewer_profile.additional_context,
            "review_count": config.generation.count,
        }

        # Save reviewers context
        reviewers_context_path = Path(job_paths["contexts"]) / "reviewers-context.json"
        with open(reviewers_context_path, "w") as f:
            json.dump(reviewers_context, f, indent=2)

        if convex:
            await convex.update_composition_progress(job_id, 75, "RGM")
            await convex.add_log(job_id, "INFO", "RGM", "Reviewer specs configured")

        # ========================================
        # Phase 3: ACM - Attributes Composition Module
        # ========================================
        if convex:
            await convex.update_composition_progress(job_id, 75, "ACM")
            await convex.add_log(
                job_id,
                "INFO",
                "ACM",
                f"Composing attributes (polarity: {config.attributes_profile.polarity.positive:.0%} pos)",
            )

        # Attributes context contains ONLY the specs/distribution
        attributes_context = {
            "polarity": {
                "positive": config.attributes_profile.polarity.positive,
                "neutral": config.attributes_profile.polarity.neutral,
                "negative": config.attributes_profile.polarity.negative,
            },
            "noise": {
                "typo_rate": config.attributes_profile.noise.typo_rate,
                "colloquialism": config.attributes_profile.noise.colloquialism,
                "grammar_errors": config.attributes_profile.noise.grammar_errors,
                "preset": config.attributes_profile.noise.preset,
            },
            "length_range": config.attributes_profile.length_range,
            "edge_lengths": getattr(config.attributes_profile, 'edge_lengths', None) and getattr(config.attributes_profile, 'edge_lengths').model_dump(),
        }

        # Save attributes context
        attributes_context_path = Path(job_paths["contexts"]) / "attributes-context.json"
        with open(attributes_context_path, "w") as f:
            json.dump(attributes_context, f, indent=2)

        if convex:
            await convex.update_composition_progress(job_id, 80, "ACM")
            await convex.add_log(job_id, "INFO", "ACM", "Attributes context saved")

        # Persona generation, writing patterns, and structure variants are now
        # generated per-target during the Generation phase (see execute_generation).
        # This keeps Composition fully target-size-agnostic for multi-target support.

        # ========================================
        # Complete composition
        # ========================================
        if convex:
            await convex.complete_composition(job_id)
            await convex.add_log(
                job_id,
                "INFO",
                "Composition",
                f"Composition complete. Ready for generation.",
            )

        return {
            "status": "composed",
            "jobId": job_id,
            "jobDir": job_paths["root"],
            "paths": job_paths,
        }

    except Exception as e:
        error_msg = str(e)
        if convex:
            await convex.add_log(job_id, "ERROR", "Composition", f"Composition failed: {error_msg}")
            await convex.fail_job(job_id, error_msg)
        raise


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "name": "CERA API",
        "version": "0.1.0",
        "status": "running",
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


class ValidateApiKeyRequest(BaseModel):
    apiKey: str


@app.post("/api/validate/openrouter")
async def validate_openrouter(request: ValidateApiKeyRequest):
    """Validate OpenRouter API key by making a test request."""
    if not request.apiKey:
        return {"valid": False, "error": "No API key provided"}

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                "https://openrouter.ai/api/v1/models",
                headers={"Authorization": f"Bearer {request.apiKey}"},
            )
            if response.status_code == 200:
                return {"valid": True}
            elif response.status_code == 401:
                return {"valid": False, "error": "Invalid API key"}
            else:
                return {"valid": False, "error": f"HTTP {response.status_code}"}
    except httpx.TimeoutException:
        return {"valid": False, "error": "Request timeout"}
    except Exception as e:
        return {"valid": False, "error": str(e)}


@app.post("/api/validate/tavily")
async def validate_tavily(request: ValidateApiKeyRequest):
    """Validate Tavily API key by making a test request."""
    if not request.apiKey:
        return {"valid": False, "error": "No API key provided"}

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Tavily uses POST with api_key in body
            response = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": request.apiKey,
                    "query": "test",
                    "max_results": 1,
                },
            )
            if response.status_code == 200:
                return {"valid": True}
            elif response.status_code == 401 or response.status_code == 403:
                return {"valid": False, "error": "Invalid API key"}
            else:
                # Check if error message indicates invalid key
                try:
                    data = response.json()
                    if "error" in data:
                        return {"valid": False, "error": data["error"]}
                except Exception:
                    pass
                return {"valid": False, "error": f"HTTP {response.status_code}"}
    except httpx.TimeoutException:
        return {"valid": False, "error": "Request timeout"}
    except Exception as e:
        return {"valid": False, "error": str(e)}


@app.post("/api/run-job")
async def run_job(request: JobRequest, background_tasks: BackgroundTasks):
    """Start a pipeline job in the background (legacy endpoint - use /api/run-pipeline instead)."""
    background_tasks.add_task(
        execute_pipeline,
        request.jobId,
        request.jobName,
        request.config,
        ["composition", "generation", "evaluation"],  # Default phases
        request.apiKey,
        None,  # tavily_api_key
        request.jobsDirectory,
        request.convexUrl,
        request.convexToken,
        None,  # evaluation_config
        None,  # dataset_file
        None,  # reused_from_job_dir
    )
    return {"status": "started", "jobId": request.jobId}


@app.post("/api/compose-job")
async def compose_job(request: Union[CompositionRequest, CompositionRequestV2]):
    """
    Execute composition phase synchronously and return when complete.

    This creates job directories and generates composition context files:
    - subject-context.json
    - reviewers-context.json
    - attributes-context.json

    Accepts two request formats:
    - V1 (CompositionRequest): Full config with all fields
    - V2 (CompositionRequestV2): Just jobId and convex credentials (fetches rest from Convex)

    The job status will be updated to 'composed' when complete.
    """
    try:
        # Check if this is the simplified V2 request (no config field)
        if isinstance(request, CompositionRequestV2) or not hasattr(request, 'config') or request.config is None:
            # V2 format - fetch job and settings from Convex
            convex = ConvexClient(request.convexUrl, request.convexToken, pocketbase_url=os.environ.get("POCKETBASE_URL"))

            # Fetch job data
            job = await convex.get_job(request.jobId)
            if not job:
                raise HTTPException(status_code=404, detail="Job not found in Convex")

            # Fetch settings
            settings = await convex.get_settings()
            if not settings:
                raise HTTPException(status_code=404, detail="Settings not found in Convex")

            if not settings.get("openrouterApiKey"):
                raise HTTPException(status_code=400, detail="OpenRouter API key not configured in settings")

            # Convert Convex job config to our format
            job_name = job.get("name", "")
            config = job.get("config", {})

            # Build the config object from Convex data
            result = await execute_composition(
                request.jobId,
                job_name,
                JobConfig(
                    subject_profile=SubjectProfile(**config.get("subject_profile", {})),
                    reviewer_profile=ReviewerProfile(**config.get("reviewer_profile", {})),
                    attributes_profile=AttributesProfile(
                        polarity=PolarityConfig(**config.get("attributes_profile", {}).get("polarity", {})),
                        noise=NoiseConfig(**config.get("attributes_profile", {}).get("noise", {})),
                        length_range=config.get("attributes_profile", {}).get("length_range", [2, 5]),
                        temp_range=config.get("attributes_profile", {}).get("temp_range", [0.85, 0.95]),
                        cap_weights=config.get("attributes_profile", {}).get("cap_weights", None),
                    ),
                    generation=GenerationConfig(**config.get("generation", {})),
                ),
                settings.get("openrouterApiKey"),
                settings.get("tavilyApiKey"),
                settings.get("jobsDirectory", "./jobs"),
                request.convexUrl,
                request.convexToken,
            )
            return result
        else:
            # V1 format - use provided config
            result = await execute_composition(
                request.jobId,
                request.jobName,
                request.config,
                request.apiKey,
                request.tavilyApiKey,
                request.jobsDirectory,
                request.convexUrl,
                request.convexToken,
            )
            return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Composition failed: {str(e)}")


class CompositionRequestSimple(BaseModel):
    """
    Simplified request for composition - all data provided by the Convex action.
    No Convex callbacks - just file operations.
    """
    jobId: str
    jobName: str
    config: JobConfig
    apiKey: str
    tavilyApiKey: Optional[str] = None
    jobsDirectory: str = "./jobs"
    # Convex credentials - currently unused but passed for future progress updates
    convexUrl: Optional[str] = None
    convexToken: Optional[str] = None


async def execute_composition_simple(
    job_id: str,
    job_name: str,
    config: JobConfig,
    api_key: str,
    tavily_api_key: Optional[str],
    jobs_directory: str,
    convex_url: Optional[str] = None,
    convex_token: Optional[str] = None,
    usage_tracker=None,
) -> dict:
    """
    Execute composition phase with optional Convex progress updates.
    Creates files and returns results.
    If convex_url and convex_token are provided, sends real-time progress updates.
    """
    import json
    from pathlib import Path
    from datetime import datetime

    from cera.pipeline.composition.sil import (
        SubjectIntelligenceLayer,
        MAVConfig as SILMAVConfig,
    )

    # Initialize Convex client for real-time updates
    convex = None
    if convex_url and convex_token:
        convex = ConvexClient(url=convex_url, token=convex_token, pocketbase_url=os.environ.get("POCKETBASE_URL"))

    # Create job directory structure first (needed for start_composing)
    job_paths = create_job_directory(jobs_directory, job_id, job_name)

    # Ensure job is in "composing" status before sending progress updates
    # This fixes race condition where Convex action's startComposing may not have completed
    if convex:
        await convex.start_composing(job_id, job_paths["root"])
        print(f"[Composition] Job status set to 'composing'")

    async def log_progress(phase: str, message: str, progress: Optional[int] = None, level: str = "INFO"):
        """Log to console and optionally to Convex."""
        print(f"[{phase}] {message}")
        if convex:
            await convex.add_log(job_id, level, phase, message)
            if progress is not None:
                await convex.update_composition_progress(job_id, progress, phase)

    # ========================================
    # Phase 1: SIL - Subject Intelligence Layer (with MAV)
    # ========================================

    # Configure MAV if enabled
    mav_enabled = config.subject_profile.mav.enabled if config.subject_profile.mav else False
    # Ablation override: force MAV off if ablation says so
    if config.ablation and not config.ablation.mav_enabled:
        mav_enabled = False
    mav_models = config.subject_profile.mav.models if config.subject_profile.mav else []
    mav_threshold = config.subject_profile.mav.similarity_threshold if config.subject_profile.mav else 0.85
    mav_answer_threshold = getattr(config.subject_profile.mav, 'answer_threshold', 0.80) if config.subject_profile.mav else 0.80
    mav_max_queries = getattr(config.subject_profile.mav, 'max_queries', 30) if config.subject_profile.mav else 30

    sil_mav_config = SILMAVConfig(
        enabled=mav_enabled and len(mav_models) >= 2,
        models=mav_models,
        similarity_threshold=mav_threshold,
        answer_threshold=mav_answer_threshold,
        max_queries=mav_max_queries,
    )

    # Create async log callback that sends SIL logs to Convex (with optional progress)
    async def sil_log_callback(level: str, phase: str, message: str, progress: int = None):
        if convex:
            await convex.add_log(job_id, level, phase, message)
            if progress is not None:
                await convex.update_composition_progress(job_id, progress, phase)

    # Initialize SIL with MAV config and log callback
    sil_enabled = config.ablation.sil_enabled if config.ablation else True
    sil = SubjectIntelligenceLayer(
        api_key=api_key,
        mav_config=sil_mav_config,
        tavily_api_key=tavily_api_key,
        log_callback=sil_log_callback if convex else None,
        usage_tracker=usage_tracker,
        sil_enabled=sil_enabled,
    )

    # Gather intelligence (this runs MAV if enabled)
    if not sil_enabled:
        await log_progress("SIL", "SIL disabled — using LLM parametric knowledge only")
    await log_progress("SIL", f"Starting gather_intelligence for: {config.subject_profile.query}", 5)
    await log_progress("SIL", f"MAV config: enabled={sil_mav_config.enabled}, models={sil_mav_config.models}")

    mav_result = await sil.gather_intelligence(
        query=config.subject_profile.query,
        region=config.subject_profile.region,
        domain=config.subject_profile.resolved_domain,
        sentiment_depth=config.subject_profile.sentiment_depth,
        additional_context=getattr(config.subject_profile, 'additional_context', None),
        aspect_categories=getattr(config.subject_profile, 'aspect_categories', None),
    )

    await log_progress("SIL", f"gather_intelligence completed", 25)
    await log_progress("SIL", f"model_data count: {len(mav_result.model_data)}")
    await log_progress("SIL", f"facts: extracted={mav_result.total_facts_extracted}, verified={mav_result.facts_verified}")

    # Build subject context from MAV result
    ctx = mav_result.context
    subject_context = {
        "query": ctx.subject,
        "region": config.subject_profile.region,
        "domain": config.subject_profile.resolved_domain,
        "sentiment_depth": config.subject_profile.sentiment_depth,
        "characteristics": ctx.features,
        "positives": ctx.pros,
        "negatives": ctx.cons,
        "use_cases": ctx.use_cases,
        "availability": ctx.availability,
        "mav_verified": ctx.mav_verified,
        "search_sources": ctx.search_sources,
        "mav_stats": {
            "total_facts_extracted": mav_result.total_facts_extracted,
            "facts_verified": mav_result.facts_verified,
            "facts_rejected": mav_result.facts_rejected,
            "verification_rate": (
                mav_result.facts_verified / mav_result.total_facts_extracted
                if mav_result.total_facts_extracted > 0 else 0
            ),
        },
    }

    # Save subject context
    subject_context_path = Path(job_paths["contexts"]) / "subject-context.json"
    with open(subject_context_path, "w") as f:
        json.dump(subject_context, f, indent=2)

    # Save verified facts (entity-clustered) if available
    if mav_result.verified_facts:
        verified_facts_path = Path(job_paths["contexts"]) / "verified-facts.json"
        with open(verified_facts_path, "w") as f:
            json.dump(mav_result.verified_facts, f, indent=2)
        n_ent = mav_result.verified_facts.get("total_entities", 0)
        await log_progress("SIL", f"Saved verified-facts.json with {n_ent} entities")

    # ========================================
    # Save MAV raw data to mavs/ directory
    # ========================================
    mavs_dir = Path(job_paths["mavs"])
    timestamp = datetime.utcnow().isoformat() + "Z"

    await log_progress("MAV", f"Saving raw data to: {mavs_dir}", 35)
    await log_progress("MAV", f"Processing {len(mav_result.model_data)} model results")

    for model_data in mav_result.model_data:
        await log_progress("MAV", f"Model: {model_data.model}, error: {model_data.error}")
        if model_data.error:
            # Skip failed models but log the error
            await log_progress("MAV", f"Skipping {model_data.model} due to error: {model_data.error}", level="WARNING")
            continue

        # Create model-specific directory
        model_dir = mavs_dir / model_data.sanitized_model
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save understanding.md - How the model understood the subject
        if model_data.understanding:
            understanding_content = f"""# MAV Understanding - {model_data.model}

## Subject
{config.subject_profile.query}

## Subject Type
{model_data.understanding.subject_type}

## Relevant Aspects
{chr(10).join(f'- {aspect}' for aspect in model_data.understanding.relevant_aspects)}

## Timestamp
{timestamp}
"""
            (model_dir / "understanding.md").write_text(understanding_content, encoding="utf-8")

        # Save queries.json - Factual queries this model generated
        if model_data.queries_generated:
            queries_data = {
                "model": model_data.model,
                "subject": config.subject_profile.query,
                "queries": model_data.queries_generated,
                "count": len(model_data.queries_generated),
                "timestamp": timestamp,
            }
            (model_dir / "queries.json").write_text(json.dumps(queries_data, indent=2), encoding="utf-8")

        # Save answers.json - This model's answers to pooled queries
        if model_data.answers:
            answers_data = {
                "model": model_data.model,
                "subject": config.subject_profile.query,
                "answers": [
                    {"query_id": a.query_id, "response": a.response, "confidence": a.confidence}
                    for a in model_data.answers
                ],
                "count": len(model_data.answers),
                "timestamp": timestamp,
            }
            (model_dir / "answers.json").write_text(json.dumps(answers_data, indent=2), encoding="utf-8")

    # ========================================
    # Generate MAV Report
    # ========================================
    reports_dir = Path(job_paths["reports"])

    if mav_result.query_pool_report:
        report = mav_result.query_pool_report

        # Build MAV report JSON (query-based format)
        mav_report = {
            "generated_at": timestamp,
            "subject": config.subject_profile.query,
            "additional_context": getattr(config.subject_profile, 'additional_context', None),
            "config": {
                "models": [md.model for md in mav_result.model_data if not md.error],
                "model_count": len([md for md in mav_result.model_data if not md.error]),
                "answer_similarity_threshold": report.threshold_used,
                "max_queries": mav_max_queries,
                "consensus_method": "LLM-judged agreement voting",
            },
            "summary": {
                "total_queries_generated": report.total_queries_generated,
                "queries_after_dedup": report.queries_after_dedup,
                "queries_with_consensus": report.queries_with_consensus,
                "queries_without_consensus": report.queries_without_consensus,
                "consensus_rate": (
                    report.queries_with_consensus / report.queries_after_dedup
                    if report.queries_after_dedup > 0 else 0
                ),
                "used_fallback": report.used_fallback,
            },
            "per_query_consensus": [
                {
                    "query_id": r.query_id,
                    "query": r.query,
                    "consensus_reached": r.consensus_reached,
                    "consensus_answer": r.consensus_answer,
                    "answers": [
                        {"model": a.model, "response": a.response, "confidence": a.confidence}
                        for a in r.answers
                    ],
                    "agreeing_models": r.agreeing_models,
                    "pairwise_similarities": r.pairwise_similarities,
                    "agreement_count": r.agreement_count,
                    "agreement_votes": r.agreement_votes,
                    "total_points": r.total_points,
                    "points_by_source": r.points_by_source,
                }
                for r in report.per_query_results
            ],
            "classified_facts": {
                "characteristics": mav_result.context.features,
                "positives": mav_result.context.pros,
                "negatives": mav_result.context.cons,
                "use_cases": mav_result.context.use_cases,
            },
        }

        # Save mav-report.json to mavs/ dir (alongside per-model raw data)
        mav_report_path = mavs_dir / "mav-report.json"
        with open(mav_report_path, "w", encoding="utf-8") as f:
            json.dump(mav_report, f, indent=2)

        # Generate mav-summary.csv for paper tables
        import csv
        summary_path = mavs_dir / "mav-summary.csv"
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", "Value"])
            writer.writerow(["subject", config.subject_profile.query])
            writer.writerow(["models_used", ",".join(md.model for md in mav_result.model_data if not md.error)])
            writer.writerow(["total_queries_generated", report.total_queries_generated])
            writer.writerow(["queries_after_dedup", report.queries_after_dedup])
            writer.writerow(["queries_with_consensus", report.queries_with_consensus])
            writer.writerow(["consensus_rate", f"{mav_report['summary']['consensus_rate']:.4f}"])
            writer.writerow(["answer_threshold", report.threshold_used])
            writer.writerow(["consensus_method", "LLM-judged agreement voting"])
            writer.writerow(["used_fallback", str(report.used_fallback)])

        await log_progress("MAV", "Generated reports: mav-report.json, mav-summary.csv", 50)
    else:
        await log_progress("MAV", "No query pool report available - skipping report generation", 50, level="WARNING")

    # ========================================
    # Phase 2: RGM - Reviewer Generation Module
    # ========================================
    await log_progress("RGM", "Building reviewer specifications...", 60)

    # Check ablation settings for age and sex
    age_enabled = getattr(config.ablation, 'age_enabled', True) if config.ablation else True
    sex_enabled = getattr(config.ablation, 'sex_enabled', True) if config.ablation else True

    # Reviewers context contains ONLY the specs/distribution, not per-review assignments
    # Per-review assignments are generated during the generation phase
    # Respect ablation settings: None for age_range if disabled, 100% unspecified for sex if disabled
    reviewers_context = {
        "age_range": config.reviewer_profile.age_range if age_enabled else None,
        "sex_distribution": config.reviewer_profile.sex_distribution if sex_enabled else {"male": 0.0, "female": 0.0, "unspecified": 1.0},
        "additional_context": config.reviewer_profile.additional_context,
        "review_count": config.generation.count,
    }

    # Save reviewers context
    reviewers_context_path = Path(job_paths["contexts"]) / "reviewers-context.json"
    with open(reviewers_context_path, "w") as f:
        json.dump(reviewers_context, f, indent=2)

    if age_enabled:
        age_info = f"age: {config.reviewer_profile.age_range}"
    else:
        age_info = "age: disabled (ablation)"
    await log_progress("RGM", f"Reviewer specs configured ({age_info})", 75)

    # ========================================
    # Phase 3: ACM - Attributes Composition Module
    # ========================================
    await log_progress("ACM", "Configuring attributes distribution...", 80)

    # Attributes context contains ONLY the specs/distribution, not per-review assignments
    # Per-review polarity/length assignments are generated during the generation phase
    attributes_context = {
        "polarity": {
            "positive": config.attributes_profile.polarity.positive,
            "neutral": config.attributes_profile.polarity.neutral,
            "negative": config.attributes_profile.polarity.negative,
        },
        "noise": {
            "typo_rate": config.attributes_profile.noise.typo_rate,
            "colloquialism": config.attributes_profile.noise.colloquialism,
            "grammar_errors": config.attributes_profile.noise.grammar_errors,
            "preset": config.attributes_profile.noise.preset,
        },
        "length_range": config.attributes_profile.length_range,
        "edge_lengths": getattr(config.attributes_profile, 'edge_lengths', None) and getattr(config.attributes_profile, 'edge_lengths').model_dump(),
    }

    # Save attributes context
    attributes_context_path = Path(job_paths["contexts"]) / "attributes-context.json"
    with open(attributes_context_path, "w") as f:
        json.dump(attributes_context, f, indent=2)

    polarity = config.attributes_profile.polarity
    await log_progress("ACM", f"Polarity set: {polarity.positive}%+ / {polarity.neutral}%~ / {polarity.negative}%-", 80)

    # Persona generation, writing patterns, and structure variants are now
    # generated per-target during the Generation phase (see execute_generation).
    # This keeps Composition fully target-size-agnostic for multi-target support.

    await log_progress("ACM", "Composition complete!", 100)

    return {
        "status": "composed",
        "jobId": job_id,
        "jobDir": job_paths["root"],
        "paths": job_paths,
        # Return contexts so Convex action can store them in the database
        "subjectContext": subject_context,
        "reviewerContext": reviewers_context,
        "attributesContext": attributes_context,
    }


@app.post("/api/compose-job-simple")
async def compose_job_simple(request: CompositionRequestSimple):
    """
    Execute composition phase with optional Convex progress updates.

    This endpoint is called by Convex actions which provide all needed data.
    If convex_url and convex_token are provided, sends real-time progress updates.
    """
    try:
        result = await execute_composition_simple(
            request.jobId,
            request.jobName,
            request.config,
            request.apiKey,
            request.tavilyApiKey,
            request.jobsDirectory,
            convex_url=request.convexUrl,
            convex_token=request.convexToken,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Composition failed: {str(e)}")


@app.get("/api/status")
async def get_status():
    """Get API server status."""
    return {
        "status": "running",
        "version": "0.1.0",
        "capabilities": [
            "generate",
            "evaluate",
            "export",
        ],
    }


@app.get("/api/prompts/{category}/{name}")
async def get_prompt(category: str, name: str):
    """Get a prompt template by category and name."""
    from cera.prompts import load_prompt

    try:
        content = load_prompt(category, name)
        return {"category": category, "name": name, "content": content}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Prompt not found: {category}/{name}")


@app.get("/api/prompts")
async def list_prompts():
    """List all available prompt templates."""
    from cera.prompts import get_available_prompts

    return {"prompts": get_available_prompts()}


@app.get("/api/config/ui-constraints")
async def get_ui_constraints():
    """Get UI constraints for form validation."""
    import json
    from pathlib import Path

    config_path = Path(__file__).parent / "config" / "ui-constraints.json"
    if not config_path.exists():
        raise HTTPException(status_code=404, detail="UI constraints config not found")

    with open(config_path) as f:
        return json.load(f)


@app.get("/api/aspect-categories")
async def get_aspect_categories():
    """Get built-in aspect category presets by domain."""
    import json
    from pathlib import Path

    config_path = Path(__file__).parent / "config" / "aspect-categories.json"
    if not config_path.exists():
        raise HTTPException(status_code=404, detail="Aspect categories config not found")

    with open(config_path) as f:
        return json.load(f)


@app.get("/api/aspect-category-presets")
async def get_aspect_category_presets():
    """List all saved aspect category preset files."""
    from pathlib import Path

    presets_dir = Path(__file__).parent / "config" / "custom-aspect-categories"
    presets = []
    if presets_dir.exists():
        for f in presets_dir.glob("*.json"):
            import json
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                presets.append({
                    "filename": f.name,
                    "name": data.get("name", f.stem),
                    "source": data.get("source", ""),
                    "categories": data.get("categories", []),
                })
            except Exception:
                continue
    return presets


class ExtractAspectCategoriesRequest(BaseModel):
    content: str
    format: str = "auto"  # auto, jsonl, csv, xml


@app.post("/api/extract-aspect-categories")
async def extract_aspect_categories(request: ExtractAspectCategoriesRequest):
    """Extract unique aspect categories from an uploaded dataset or category list file."""
    import csv as csv_module
    import io
    import re

    categories = set()
    content = request.content
    fmt = request.format

    # Auto-detect format
    if fmt == "auto":
        stripped = content.strip()
        first_line = stripped.split("\n")[0].strip() if stripped else ""

        if stripped.startswith("<?xml") or stripped.startswith("<Reviews") or stripped.startswith("<AspectCategories"):
            fmt = "xml"
        elif first_line.startswith("{"):
            # First line is JSON object - treat as JSONL
            fmt = "jsonl"
        elif "review_id,sentence_id" in content[:200] or first_line.lower() == "category" or ",category" in first_line.lower():
            # CSV with header row
            fmt = "csv"
        else:
            fmt = "jsonl"

    if fmt == "xml":
        # Parse XML for category attributes (SemEval format: category="...")
        for match in re.finditer(r'category="([^"]+)"', content):
            categories.add(match.group(1))
        # Also support exported format: <Category>...</Category>
        for match in re.finditer(r'<Category>([^<]+)</Category>', content):
            categories.add(match.group(1))
    elif fmt == "csv":
        reader = csv_module.DictReader(io.StringIO(content))
        for row in reader:
            if "category" in row and row["category"]:
                categories.add(row["category"])
    elif fmt == "jsonl":
        import json
        for line in content.strip().split("\n"):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                # Support full review format: sentences[].opinions[].category
                for sent in entry.get("sentences", []):
                    for op in sent.get("opinions", []):
                        if "category" in op:
                            categories.add(op["category"])
                # Support simple exported format: {"category": "..."}
                if "category" in entry and isinstance(entry["category"], str):
                    categories.add(entry["category"])
            except Exception:
                continue

    return {"categories": sorted(categories), "count": len(categories)}


class SaveAspectPresetRequest(BaseModel):
    name: str
    source: str = ""
    categories: list[str]


@app.post("/api/save-aspect-preset")
async def save_aspect_preset(request: SaveAspectPresetRequest):
    """Save extracted aspect categories as a named preset."""
    import json
    from pathlib import Path

    presets_dir = Path(__file__).parent / "config" / "custom-aspect-categories"
    presets_dir.mkdir(parents=True, exist_ok=True)

    # Sanitize filename
    filename = re.sub(r'[^\w\-]', '-', request.name.lower().strip()) + ".json"
    filepath = presets_dir / filename

    data = {
        "name": request.name,
        "source": request.source,
        "categories": request.categories,
    }
    filepath.write_text(json.dumps(data, indent=2), encoding="utf-8")

    return {"status": "saved", "filename": filename, "path": str(filepath)}


@app.delete("/api/delete-aspect-preset/{filename}")
async def delete_aspect_preset(filename: str):
    """Delete a custom aspect category preset."""
    from pathlib import Path

    presets_dir = Path(__file__).parent / "config" / "custom-aspect-categories"
    filepath = presets_dir / filename

    # Security: ensure path is within presets directory
    try:
        filepath = filepath.resolve()
        if not str(filepath).startswith(str(presets_dir.resolve())):
            raise HTTPException(status_code=400, detail="Invalid filename")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid filename")

    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Preset not found")

    filepath.unlink()
    return {"status": "deleted", "filename": filename}


class ConvertFormatRequest(BaseModel):
    content: str
    source_format: str  # jsonl, csv, xml
    target_format: str  # jsonl, csv, xml


@app.post("/api/convert-format")
async def convert_format(request: ConvertFormatRequest):
    """Convert dataset between JSONL, CSV, and SemEval XML formats."""
    import csv as csv_module
    import io
    import re

    # Parse source format into unified internal structure
    reviews = []  # List of {id, sentences: [{id, text, opinions: [{target, category, polarity, from, to}]}]}

    if request.source_format == "jsonl":
        import json
        for line in request.content.strip().split("\n"):
            if not line.strip():
                continue
            try:
                reviews.append(json.loads(line))
            except Exception:
                continue

    elif request.source_format == "csv":
        reader = csv_module.DictReader(io.StringIO(request.content))
        # Group by review_id
        review_map = {}
        for row in reader:
            rid = row.get("review_id", "0")
            sid = row.get("sentence_id", f"{rid}:0")
            if rid not in review_map:
                review_map[rid] = {"id": rid, "sentences": {}}
            if sid not in review_map[rid]["sentences"]:
                review_map[rid]["sentences"][sid] = {"id": sid, "text": row.get("text", ""), "opinions": []}
            review_map[rid]["sentences"][sid]["opinions"].append({
                "target": row.get("target", "NULL"),
                "category": row.get("category", ""),
                "polarity": row.get("polarity", "neutral"),
                "from": int(row.get("from", 0)),
                "to": int(row.get("to", 0)),
            })
        for rid, rdata in review_map.items():
            reviews.append({"id": rid, "sentences": list(rdata["sentences"].values())})

    elif request.source_format == "xml":
        # Parse SemEval XML
        review_blocks = re.findall(r'<Review\s[^>]*rid="([^"]*)"[^>]*>(.*?)</Review>', request.content, re.DOTALL)
        for rid, block in review_blocks:
            sentences = []
            sent_blocks = re.findall(r'<sentence\s[^>]*id="([^"]*)"[^>]*>(.*?)</sentence>', block, re.DOTALL)
            if sent_blocks:
                for sid, sblock in sent_blocks:
                    text_match = re.search(r'<text>(.*?)</text>', sblock, re.DOTALL)
                    text = text_match.group(1) if text_match else ""
                    # Unescape XML
                    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"')
                    opinions = []
                    for op_match in re.finditer(r'<Opinion\s+([^/]*)/>', sblock):
                        attrs = op_match.group(1)
                        target = re.search(r'target="([^"]*)"', attrs)
                        cat = re.search(r'category="([^"]*)"', attrs)
                        pol = re.search(r'polarity="([^"]*)"', attrs)
                        # Support both standard (from/to) and hotel format (target_from/target_to)
                        frm = re.search(r'\bfrom="([^"]*)"', attrs) or re.search(r'target_from="([^"]*)"', attrs)
                        to = re.search(r'\bto="([^"]*)"', attrs) or re.search(r'target_to="([^"]*)"', attrs)
                        opinions.append({
                            "target": target.group(1) if target else "NULL",
                            "category": cat.group(1) if cat else "",
                            "polarity": pol.group(1) if pol else "neutral",
                            "from": int(frm.group(1)) if frm else 0,
                            "to": int(to.group(1)) if to else 0,
                        })
                    sentences.append({"id": sid, "text": text, "opinions": opinions})
            reviews.append({"id": rid, "sentences": sentences})

    if not reviews:
        raise HTTPException(status_code=400, detail="No reviews parsed from source content")

    # Convert to target format
    output = ""

    if request.target_format == "jsonl":
        import json
        lines = []
        for review in reviews:
            lines.append(json.dumps(review, ensure_ascii=False))
        output = "\n".join(lines) + "\n"

    elif request.target_format == "csv":
        sio = io.StringIO()
        writer = csv_module.writer(sio)
        writer.writerow(["review_id", "sentence_id", "text", "target", "category", "polarity", "from", "to"])
        for review in reviews:
            for sent in review.get("sentences", []):
                for op in sent.get("opinions", []):
                    writer.writerow([
                        review["id"],
                        sent.get("id", ""),
                        sent["text"],
                        op.get("target", "NULL"),
                        op.get("category", ""),
                        op.get("polarity", "neutral"),
                        op.get("from", 0),
                        op.get("to", 0),
                    ])
        output = sio.getvalue()

    elif request.target_format == "xml":
        xml_lines = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>', '<Reviews>']
        for review in reviews:
            xml_lines.append(f'  <Review rid="{review["id"]}">')
            xml_lines.append('    <sentences>')
            for sent in review.get("sentences", []):
                text_escaped = sent["text"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
                xml_lines.append(f'      <sentence id="{sent.get("id", "")}">')
                xml_lines.append(f'        <text>{text_escaped}</text>')
                xml_lines.append('        <Opinions>')
                for op in sent.get("opinions", []):
                    target_escaped = op.get("target", "NULL").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
                    xml_lines.append(f'          <Opinion target="{target_escaped}" category="{op.get("category", "")}" polarity="{op.get("polarity", "neutral")}" from="{op.get("from", 0)}" to="{op.get("to", 0)}" />')
                xml_lines.append('        </Opinions>')
                xml_lines.append('      </sentence>')
            xml_lines.append('    </sentences>')
            xml_lines.append('  </Review>')
        xml_lines.append('</Reviews>')
        output = "\n".join(xml_lines) + "\n"

    return {"content": output, "format": request.target_format, "review_count": len(reviews)}


class ConvertExplicitToImplicitRequest(BaseModel):
    content: str
    format: str = "auto"  # auto, jsonl, csv, xml


@app.post("/api/convert-explicit-to-implicit")
async def convert_explicit_to_implicit(request: ConvertExplicitToImplicitRequest):
    """Convert an explicit dataset to implicit (strip targets and offsets)."""
    import csv as csv_module
    import io
    import re

    content = request.content
    fmt = request.format

    # Auto-detect format
    if fmt == "auto":
        if content.strip().startswith("<?xml") or content.strip().startswith("<Reviews"):
            fmt = "xml"
        elif "review_id,sentence_id" in content[:200]:
            fmt = "csv"
        else:
            fmt = "jsonl"

    if fmt == "xml":
        # Replace target="..." with target="NULL" and from/to with 0
        output = re.sub(r'target="[^"]*"', 'target="NULL"', content)
        output = re.sub(r'from="\d+"', 'from="0"', output)
        output = re.sub(r'to="\d+"', 'to="0"', output)

    elif fmt == "csv":
        reader = csv_module.DictReader(io.StringIO(content))
        sio = io.StringIO()
        writer = csv_module.DictWriter(sio, fieldnames=["review_id", "sentence_id", "text", "target", "category", "polarity", "from", "to"])
        writer.writeheader()
        for row in reader:
            row["target"] = "NULL"
            row["from"] = "0"
            row["to"] = "0"
            writer.writerow(row)
        output = sio.getvalue()

    elif fmt == "jsonl":
        import json
        lines = []
        for line in content.strip().split("\n"):
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                for sent in entry.get("sentences", []):
                    for op in sent.get("opinions", []):
                        op["target"] = "NULL"
                        op["from"] = 0
                        op["to"] = 0
                lines.append(json.dumps(entry, ensure_ascii=False))
            except Exception:
                lines.append(line)
        output = "\n".join(lines) + "\n"
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {fmt}")

    return {"content": output, "format": fmt}


class MergeDatasetsRequest(BaseModel):
    contents: list[str]
    format: str  # jsonl, csv, xml


@app.post("/api/merge-datasets")
async def merge_datasets(request: MergeDatasetsRequest):
    """Merge multiple datasets of the same format into one."""
    import csv as csv_module
    import io
    import json
    import re

    if len(request.contents) < 2:
        raise HTTPException(status_code=400, detail="At least 2 files required to merge")

    fmt = request.format
    all_reviews = []

    for content in request.contents:
        if fmt == "jsonl":
            for line in content.strip().split("\n"):
                if not line.strip():
                    continue
                try:
                    all_reviews.append(json.loads(line))
                except Exception:
                    continue

        elif fmt == "csv":
            reader = csv_module.DictReader(io.StringIO(content))
            review_map = {}
            for row in reader:
                rid = row.get("review_id", "0")
                sid = row.get("sentence_id", f"{rid}:0")
                if rid not in review_map:
                    review_map[rid] = {"id": rid, "sentences": {}}
                if sid not in review_map[rid]["sentences"]:
                    review_map[rid]["sentences"][sid] = {"id": sid, "text": row.get("text", ""), "opinions": []}
                review_map[rid]["sentences"][sid]["opinions"].append({
                    "target": row.get("target", "NULL"),
                    "category": row.get("category", ""),
                    "polarity": row.get("polarity", "neutral"),
                    "from": int(row.get("from", 0)),
                    "to": int(row.get("to", 0)),
                })
            for rid, rdata in review_map.items():
                all_reviews.append({"id": rid, "sentences": list(rdata["sentences"].values())})

        elif fmt == "xml":
            review_blocks = re.findall(r'<Review\s[^>]*rid="([^"]*)"[^>]*>(.*?)</Review>', content, re.DOTALL)
            for rid, block in review_blocks:
                sentences = []
                sent_blocks = re.findall(r'<sentence\s[^>]*id="([^"]*)"[^>]*>(.*?)</sentence>', block, re.DOTALL)
                for sid, sblock in sent_blocks:
                    text_match = re.search(r'<text>(.*?)</text>', sblock, re.DOTALL)
                    text = text_match.group(1) if text_match else ""
                    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"')
                    opinions = []
                    for op_match in re.finditer(r'<Opinion\s+([^/]*)/>', sblock):
                        attrs = op_match.group(1)
                        target = re.search(r'target="([^"]*)"', attrs)
                        cat = re.search(r'category="([^"]*)"', attrs)
                        pol = re.search(r'polarity="([^"]*)"', attrs)
                        # Support both standard (from/to) and hotel format (target_from/target_to)
                        frm = re.search(r'\bfrom="([^"]*)"', attrs) or re.search(r'target_from="([^"]*)"', attrs)
                        to = re.search(r'\bto="([^"]*)"', attrs) or re.search(r'target_to="([^"]*)"', attrs)
                        opinions.append({
                            "target": target.group(1) if target else "NULL",
                            "category": cat.group(1) if cat else "",
                            "polarity": pol.group(1) if pol else "neutral",
                            "from": int(frm.group(1)) if frm else 0,
                            "to": int(to.group(1)) if to else 0,
                        })
                    sentences.append({"id": sid, "text": text, "opinions": opinions})
                all_reviews.append({"id": rid, "sentences": sentences})

    if not all_reviews:
        raise HTTPException(status_code=400, detail="No reviews parsed from source files")

    # Re-assign unique IDs to avoid duplicates
    for i, review in enumerate(all_reviews):
        review["id"] = str(i + 1)
        for j, sent in enumerate(review.get("sentences", [])):
            sent["id"] = f"{review['id']}:{j}"

    # Output in the same format
    output = ""
    if fmt == "jsonl":
        lines = [json.dumps(r, ensure_ascii=False) for r in all_reviews]
        output = "\n".join(lines) + "\n"

    elif fmt == "csv":
        sio = io.StringIO()
        writer = csv_module.writer(sio)
        writer.writerow(["review_id", "sentence_id", "text", "target", "category", "polarity", "from", "to"])
        for review in all_reviews:
            for sent in review.get("sentences", []):
                for op in sent.get("opinions", []):
                    writer.writerow([
                        review["id"],
                        sent.get("id", ""),
                        sent["text"],
                        op.get("target", "NULL"),
                        op.get("category", ""),
                        op.get("polarity", "neutral"),
                        op.get("from", 0),
                        op.get("to", 0),
                    ])
        output = sio.getvalue()

    elif fmt == "xml":
        xml_lines = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>', '<Reviews>']
        for review in all_reviews:
            xml_lines.append(f'  <Review rid="{review["id"]}">')
            xml_lines.append('    <sentences>')
            for sent in review.get("sentences", []):
                text_escaped = sent["text"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
                xml_lines.append(f'      <sentence id="{sent.get("id", "")}">')
                xml_lines.append(f'        <text>{text_escaped}</text>')
                xml_lines.append('        <Opinions>')
                for op in sent.get("opinions", []):
                    target_escaped = op.get("target", "NULL").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
                    xml_lines.append(f'          <Opinion target="{target_escaped}" category="{op.get("category", "")}" polarity="{op.get("polarity", "neutral")}" from="{op.get("from", 0)}" to="{op.get("to", 0)}" />')
                xml_lines.append('        </Opinions>')
                xml_lines.append('      </sentence>')
            xml_lines.append('    </sentences>')
            xml_lines.append('  </Review>')
        xml_lines.append('</Reviews>')
        output = "\n".join(xml_lines) + "\n"

    return {
        "content": output,
        "format": fmt,
        "total_records": len(all_reviews),
        "files_merged": len(request.contents),
    }


class SubsampleDatasetRequest(BaseModel):
    content: str
    format: str  # jsonl, csv, xml
    mode: str  # "equal", "custom", or "sample"
    num_splits: int | None = 2
    split_sizes: list[int] | None = None
    set_sizes: list[int] | None = None  # For "sample" mode
    unit: str | None = "review"  # "review" or "sentence"
    name_prefix: str | None = None  # File name prefix for sample mode
    name_postfix: str | None = None  # File name postfix for sample mode


@app.post("/api/subsample-dataset")
async def subsample_dataset(request: SubsampleDatasetRequest):
    """Split a dataset into smaller non-overlapping subsets."""
    import csv as csv_module
    import io
    import json
    import random
    import re

    content = request.content
    fmt = request.format

    # Parse the dataset
    reviews = []

    if fmt == "jsonl":
        for line in content.strip().split("\n"):
            if not line.strip():
                continue
            try:
                reviews.append(json.loads(line))
            except Exception:
                continue

    elif fmt == "csv":
        reader = csv_module.DictReader(io.StringIO(content))
        review_map = {}
        for row in reader:
            rid = row.get("review_id", "0")
            sid = row.get("sentence_id", f"{rid}:0")
            if rid not in review_map:
                review_map[rid] = {"id": rid, "sentences": {}}
            if sid not in review_map[rid]["sentences"]:
                review_map[rid]["sentences"][sid] = {"id": sid, "text": row.get("text", ""), "opinions": []}
            review_map[rid]["sentences"][sid]["opinions"].append({
                "target": row.get("target", "NULL"),
                "category": row.get("category", ""),
                "polarity": row.get("polarity", "neutral"),
                "from": int(row.get("from", 0)),
                "to": int(row.get("to", 0)),
            })
        for rid, rdata in review_map.items():
            reviews.append({"id": rid, "sentences": list(rdata["sentences"].values())})

    elif fmt == "xml":
        review_blocks = re.findall(r'<Review\s[^>]*rid="([^"]*)"[^>]*>(.*?)</Review>', content, re.DOTALL)
        for rid, block in review_blocks:
            sentences = []
            sent_blocks = re.findall(r'<sentence\s[^>]*id="([^"]*)"[^>]*>(.*?)</sentence>', block, re.DOTALL)
            for sid, sblock in sent_blocks:
                text_match = re.search(r'<text>(.*?)</text>', sblock, re.DOTALL)
                text = text_match.group(1) if text_match else ""
                text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">").replace("&quot;", '"')
                opinions = []
                for op_match in re.finditer(r'<Opinion\s+([^/]*)/>', sblock):
                    attrs = op_match.group(1)
                    target = re.search(r'target="([^"]*)"', attrs)
                    cat = re.search(r'category="([^"]*)"', attrs)
                    pol = re.search(r'polarity="([^"]*)"', attrs)
                    # Support both standard (from/to) and hotel format (target_from/target_to)
                    frm = re.search(r'\bfrom="([^"]*)"', attrs) or re.search(r'target_from="([^"]*)"', attrs)
                    to = re.search(r'\bto="([^"]*)"', attrs) or re.search(r'target_to="([^"]*)"', attrs)
                    opinions.append({
                        "target": target.group(1) if target else "NULL",
                        "category": cat.group(1) if cat else "",
                        "polarity": pol.group(1) if pol else "neutral",
                        "from": int(frm.group(1)) if frm else 0,
                        "to": int(to.group(1)) if to else 0,
                    })
                sentences.append({"id": sid, "text": text, "opinions": opinions})
            reviews.append({"id": rid, "sentences": sentences})

    if not reviews:
        raise HTTPException(status_code=400, detail="No reviews parsed from content")

    total_records = len(reviews)

    def serialize_reviews(review_list: list, fmt: str) -> str:
        """Serialize a list of reviews to the given format."""
        if fmt == "jsonl":
            return "\n".join([json.dumps(r, ensure_ascii=False) for r in review_list]) + "\n"
        elif fmt == "csv":
            sio = io.StringIO()
            writer = csv_module.writer(sio)
            writer.writerow(["review_id", "sentence_id", "text", "target", "category", "polarity", "from", "to"])
            for review in review_list:
                for sent in review.get("sentences", []):
                    for op in sent.get("opinions", []):
                        writer.writerow([
                            review["id"],
                            sent.get("id", ""),
                            sent["text"],
                            op.get("target", "NULL"),
                            op.get("category", ""),
                            op.get("polarity", "neutral"),
                            op.get("from", 0),
                            op.get("to", 0),
                        ])
            return sio.getvalue()
        elif fmt == "xml":
            xml_lines = ['<?xml version="1.0" encoding="UTF-8" standalone="yes"?>', '<Reviews>']
            for review in review_list:
                xml_lines.append(f'  <Review rid="{review["id"]}">')
                xml_lines.append('    <sentences>')
                for sent in review.get("sentences", []):
                    text_escaped = sent["text"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
                    xml_lines.append(f'      <sentence id="{sent.get("id", "")}">')
                    xml_lines.append(f'        <text>{text_escaped}</text>')
                    xml_lines.append('        <Opinions>')
                    for op in sent.get("opinions", []):
                        target_escaped = op.get("target", "NULL").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
                        xml_lines.append(f'          <Opinion target="{target_escaped}" category="{op.get("category", "")}" polarity="{op.get("polarity", "neutral")}" from="{op.get("from", 0)}" to="{op.get("to", 0)}" />')
                    xml_lines.append('        </Opinions>')
                    xml_lines.append('      </sentence>')
                xml_lines.append('    </sentences>')
                xml_lines.append('  </Review>')
            xml_lines.append('</Reviews>')
            return "\n".join(xml_lines) + "\n"
        return ""

    def reassign_ids(review_list: list) -> None:
        """Re-assign sequential IDs within a list of reviews."""
        for j, review in enumerate(review_list):
            review["id"] = str(j + 1)
            for k, sent in enumerate(review.get("sentences", [])):
                sent["id"] = f"{review['id']}:{k}"

    # Count total sentences
    total_sentences = sum(len(r.get("sentences", [])) for r in reviews)

    # === Sample mode: independent random sets with overlap ===
    if request.mode == "sample":
        import copy
        set_sizes = request.set_sizes or [100]
        unit = request.unit or "review"
        prefix = request.name_prefix or "set"
        postfix = request.name_postfix or ""

        def make_set_name(target: int) -> str:
            parts = [prefix, str(target)]
            if postfix:
                parts.append(postfix)
            return "-".join(parts)

        splits = []
        for i, target_size in enumerate(set_sizes):
            if unit == "sentence":
                # Pick random sentences until we hit the target count
                # Build a flat list of (review_index, sentence_index)
                all_sentences = []
                for ri, review in enumerate(reviews):
                    for si in range(len(review.get("sentences", []))):
                        all_sentences.append((ri, si))

                if target_size > len(all_sentences):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Set {i+1} requests {target_size} sentences but only {len(all_sentences)} available"
                    )

                # Random sample of sentence indices
                sampled = random.sample(all_sentences, target_size)

                # Group by review to reconstruct partial reviews
                review_sentences: dict[int, list[int]] = {}
                for ri, si in sampled:
                    review_sentences.setdefault(ri, []).append(si)

                # Build review list with only the selected sentences
                set_reviews = []
                for ri in sorted(review_sentences.keys()):
                    original = reviews[ri]
                    orig_sents = original.get("sentences", [])
                    partial = copy.deepcopy(original)
                    partial["sentences"] = [copy.deepcopy(orig_sents[si]) for si in sorted(review_sentences[ri])]
                    set_reviews.append(partial)

                reassign_ids(set_reviews)
                actual_sentences = sum(len(r.get("sentences", [])) for r in set_reviews)
                splits.append({
                    "name": make_set_name(target_size),
                    "content": serialize_reviews(set_reviews, fmt),
                    "count": len(set_reviews),
                    "sentence_count": actual_sentences,
                })
            else:
                # By review: simple random sample
                if target_size > total_records:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Set {i+1} requests {target_size} reviews but only {total_records} available"
                    )
                sampled_reviews = copy.deepcopy(random.sample(reviews, target_size))
                reassign_ids(sampled_reviews)
                splits.append({
                    "name": make_set_name(target_size),
                    "content": serialize_reviews(sampled_reviews, fmt),
                    "count": len(sampled_reviews),
                })

        return {
            "splits": splits,
            "format": fmt,
            "total_records": total_records,
            "total_sentences": total_sentences,
        }

    # === Split modes (equal / custom): non-overlapping ===
    # Shuffle for random distribution
    random.shuffle(reviews)

    # Determine split sizes
    if request.mode == "equal":
        num_splits = request.num_splits or 2
        base_size = total_records // num_splits
        remainder = total_records % num_splits
        sizes = [base_size + (1 if i < remainder else 0) for i in range(num_splits)]
    else:  # custom
        sizes = request.split_sizes or [100, 100]
        if sum(sizes) > total_records:
            raise HTTPException(status_code=400, detail=f"Requested {sum(sizes)} records but only {total_records} available")

    # Create splits
    splits = []
    offset = 0
    for i, size in enumerate(sizes):
        split_reviews = reviews[offset:offset + size]
        offset += size
        reassign_ids(split_reviews)
        splits.append({
            "name": f"split-{i + 1}",
            "content": serialize_reviews(split_reviews, fmt),
            "count": len(split_reviews),
        })

    return {
        "splits": splits,
        "format": fmt,
        "total_records": total_records,
    }


class ExportRequest(BaseModel):
    dataset_path: str
    format: str = "jsonl"  # jsonl, csv, semeval
    domain: str = "restaurant"


@app.post("/api/export")
async def export_dataset(request: ExportRequest):
    """Export a dataset to the specified format."""
    from pathlib import Path
    from cera.export import DatasetExporter
    from cera.export.formats import load_dataset

    # Validate format
    valid_formats = ["jsonl", "csv", "semeval"]
    if request.format not in valid_formats:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid format '{request.format}'. Valid formats: {valid_formats}",
        )

    input_path = Path(request.dataset_path)
    if not input_path.exists():
        raise HTTPException(status_code=404, detail=f"Dataset not found: {request.dataset_path}")

    try:
        # Load dataset
        reviews = load_dataset(input_path)

        # Export to new format
        ext_map = {"jsonl": ".jsonl", "csv": ".csv", "semeval": ".xml"}
        output_path = input_path.with_suffix(ext_map[request.format])

        exporter = DatasetExporter(domain=request.domain)
        result_path = exporter.export(reviews, output_path, request.format)
        stats = exporter.get_stats(reviews)

        return {
            "status": "success",
            "output_path": str(result_path),
            "format": request.format,
            "stats": stats,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


@app.get("/api/export/{dataset_id}/{format}")
async def download_export(dataset_id: str, format: str):
    """Download a dataset in the specified format."""
    from pathlib import Path
    from fastapi.responses import FileResponse
    from cera.export import DatasetExporter
    from cera.export.formats import load_dataset

    # Validate format
    valid_formats = ["jsonl", "csv", "semeval"]
    if format not in valid_formats:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid format '{format}'. Valid formats: {valid_formats}",
        )

    # Try to find the dataset (in the output directory)
    output_dir = Path("./output")
    possible_paths = [
        output_dir / dataset_id / "reviews.jsonl",
        output_dir / f"{dataset_id}.jsonl",
        output_dir / dataset_id,
    ]

    input_path = None
    for path in possible_paths:
        if path.exists():
            input_path = path
            break

    if input_path is None:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")

    try:
        # Load dataset
        reviews = load_dataset(input_path)

        # Export to requested format
        ext_map = {"jsonl": ".jsonl", "csv": ".csv", "semeval": ".xml"}
        mime_types = {
            "jsonl": "application/jsonl",
            "csv": "text/csv",
            "semeval": "application/xml",
        }

        output_path = input_path.parent / f"{input_path.stem}_export{ext_map[format]}"

        exporter = DatasetExporter()
        result_path = exporter.export(reviews, output_path, format)

        return FileResponse(
            path=str(result_path),
            media_type=mime_types[format],
            filename=f"{dataset_id}{ext_map[format]}",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")


@app.get("/api/jobs/{job_id}/export/{report_type}")
async def export_job_report(job_id: str, report_type: str):
    """
    Export job reports as zip files.

    report_type: 'mav', 'conformity', or 'metrics'
    """
    import zipfile
    import io
    from pathlib import Path
    from fastapi.responses import StreamingResponse

    # Find job directory
    jobs_dir = Path("./jobs")
    job_dir = None
    for d in jobs_dir.iterdir():
        if d.is_dir() and d.name.startswith(job_id):
            job_dir = d
            break

    if not job_dir:
        raise HTTPException(status_code=404, detail=f"Job directory not found: {job_id}")

    # Determine which files to include based on report type
    files_to_zip = []
    zip_filename = ""

    if report_type == "mav":
        # Check mavs/ first (new), then reports/ (legacy)
        mavs_dir = job_dir / "mavs"
        reports_dir = job_dir / "reports"
        mav_report = mavs_dir / "mav-report.json" if (mavs_dir / "mav-report.json").exists() else reports_dir / "mav-report.json"
        mav_summary = mavs_dir / "mav-summary.csv" if (mavs_dir / "mav-summary.csv").exists() else reports_dir / "mav-summary.csv"
        files_to_zip = [mav_report, mav_summary]
        zip_filename = f"{job_id}-mav-report.zip"
    elif report_type == "conformity":
        # Check per-target metrics/ dirs, then reports/ (legacy)
        conf_files = []
        ds_dir = job_dir / "datasets"
        if ds_dir.exists():
            for td in ds_dir.iterdir():
                if td.is_dir() and td.name.isdigit():
                    candidate = td / "metrics" / "conformity-report.json"
                    if candidate.exists():
                        conf_files.append(candidate)
        if not conf_files:
            candidate = job_dir / "reports" / "conformity-report.json"
            if candidate.exists():
                conf_files.append(candidate)
        files_to_zip = conf_files
        zip_filename = f"{job_id}-conformity-report.zip"
    elif report_type == "metrics":
        # Check per-target metrics/ dirs, then root metrics/
        metric_files = []
        ds_dir = job_dir / "datasets"
        if ds_dir.exists():
            for td in ds_dir.iterdir():
                if td.is_dir() and td.name.isdigit():
                    m_dir = td / "metrics"
                    if m_dir.exists():
                        metric_files.extend(list(m_dir.glob("*.json")) + list(m_dir.glob("*.csv")))
        if not metric_files:
            metrics_dir = job_dir / "metrics"
            if metrics_dir.exists():
                metric_files = list(metrics_dir.glob("*.json")) + list(metrics_dir.glob("*.csv"))
        files_to_zip = metric_files
        zip_filename = f"{job_id}-mdqa-metrics.zip"
    else:
        raise HTTPException(status_code=400, detail=f"Invalid report type: {report_type}. Valid types: mav, conformity, metrics")

    # Filter to only existing files
    existing_files = [f for f in files_to_zip if f.exists()]

    if not existing_files:
        raise HTTPException(status_code=404, detail=f"No {report_type} files found for job {job_id}")

    # Create zip file in memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in existing_files:
            # Use just the filename in the zip (not full path)
            zf.write(file_path, file_path.name)

    zip_buffer.seek(0)

    return StreamingResponse(
        zip_buffer,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={zip_filename}"}
    )


# ============================================================================
# Generation Phase Endpoint
# ============================================================================


class GenerationRequest(BaseModel):
    """Request for generation phase (uses existing composition data)."""
    jobId: str
    jobDir: str  # Path to the job directory with context files
    config: JobConfig
    apiKey: str
    # Callback info for progress updates
    convexUrl: Optional[str] = None
    convexToken: Optional[str] = None


# ========================================
# AML Restructuring Helpers
# ========================================

def build_sentence_plan(num_sentences: int, aspect_pool: list[str], polarity_dist: dict) -> list[dict]:
    """Pre-assign 0-3 aspect categories and per-aspect polarity for each sentence.

    When a sentence has 2+ aspects, all aspects share the same top-level category
    (the part before '#') to produce thematically coherent sentences. For example,
    DISPLAY#GENERAL + DISPLAY#DESIGN_FEATURES can co-occur, but SHIPPING#QUALITY +
    DISPLAY#GENERAL cannot.

    Returns a list of dicts, one per sentence:
        [{"sentence": 1, "aspects": [{"category": "LAPTOP#PRICE", "polarity": "negative"}, ...]}, ...]
    Sentences with 0 aspects are contextual/transitional (empty opinions array in output).
    """
    import random

    def _pick_polarity():
        rand = random.random()
        neg_t = polarity_dist["negative"]
        neu_t = neg_t + polarity_dist["neutral"]
        return "negative" if rand < neg_t else ("neutral" if rand < neu_t else "positive")

    # Group aspects by top-level category for thematic coherence
    category_groups: dict[str, list[str]] = {}
    for asp in aspect_pool:
        top = asp.split("#")[0]
        category_groups.setdefault(top, []).append(asp)

    plan = []
    for i in range(num_sentences):
        num_aspects = random.randint(0, 3)
        if num_aspects == 0:
            plan.append({"sentence": i + 1, "aspects": []})
        else:
            # Pick a random category group, then sample within it
            group_key = random.choice(list(category_groups.keys()))
            group = category_groups[group_key]
            sampled = random.sample(group, min(num_aspects, len(group)))
            aspects = [{"category": cat, "polarity": _pick_polarity()} for cat in sampled]
            plan.append({"sentence": i + 1, "aspects": aspects})

    # Ensure at least 1 sentence has aspects (prevent all-contextual reviews)
    if all(len(s["aspects"]) == 0 for s in plan):
        idx = random.randint(0, len(plan) - 1)
        cat = random.choice(aspect_pool)
        plan[idx]["aspects"] = [{"category": cat, "polarity": _pick_polarity()}]

    return plan


def format_sentence_plan(plan: list[dict]) -> str:
    """Format a sentence plan for injection into the AML prompt."""
    lines = []
    for entry in plan:
        sent_num = entry["sentence"]
        aspects = entry["aspects"]
        if not aspects:
            lines.append(f"- Sentence {sent_num}: (contextual — personal detail or transition)")
        else:
            aspect_strs = [f"{a['category']} ({a['polarity']})" for a in aspects]
            lines.append(f"- Sentence {sent_num}: {', '.join(aspect_strs)}")
    return "\n".join(lines)


async def _generate_personas(
    persona_count: int,
    reviewer_profile,
    reviewers_context: dict,
    subject_context: dict,
    gen_model: str,
    api_key: str,
    age_enabled: bool = True,
    sex_enabled: bool = True,
    usage_tracker=None,
) -> list[dict]:
    """Generate persona objects via LLM.

    Reusable helper called from the Generation phase (per-target).
    Returns a list of persona dicts with id, name, age, sex, region, background,
    writing_tendencies, and priorities fields.
    """
    from cera.llm.openrouter import OpenRouterClient
    from cera.prompts import load_prompt, format_prompt
    from cera.pipeline.composition.rgm import ReviewerGenerationModule

    _llm = OpenRouterClient(api_key=api_key, usage_tracker=usage_tracker, component="aml.persona")

    # Pre-sample demographics via RGM
    age_range_tuple = tuple(reviewer_profile.age_range) if reviewer_profile.age_range and age_enabled else None
    sex_dist = reviewer_profile.sex_distribution if sex_enabled else {"male": 0.0, "female": 0.0, "unspecified": 1.0}
    _rgm = ReviewerGenerationModule(
        age_range=age_range_tuple,
        sex_distribution=sex_dist,
    )
    pre_sampled = _rgm.generate_profiles(persona_count)

    # Build demographics list for prompt
    demo_lines = []
    for idx, prof in enumerate(pre_sampled):
        parts = [f"Persona {idx + 1}:"]
        if prof.age is not None:
            parts.append(f"age {prof.age}")
        if prof.sex and prof.sex.lower() != "unspecified":
            parts.append(prof.sex)
        demo_lines.append(" ".join(parts))
    demographics_list = "\n".join(demo_lines)

    reviewer_ctx = reviewers_context.get("additional_context", "general consumer")
    _region = subject_context.get("region", "unknown")

    persona_prompt_template = load_prompt("composition", "personas")
    persona_prompt = format_prompt(persona_prompt_template,
        persona_count=persona_count,
        domain=subject_context.get("domain", subject_context.get("category", "general")),
        subject=subject_context.get("query", ""),
        region=_region,
        reviewer_context=reviewer_ctx,
        demographics_list=demographics_list,
    )

    PERSONA_BATCH_SIZE = 25
    all_personas = []

    for batch_start in range(0, persona_count, PERSONA_BATCH_SIZE):
        batch_end = min(batch_start + PERSONA_BATCH_SIZE, persona_count)
        batch_count = batch_end - batch_start

        if persona_count > PERSONA_BATCH_SIZE:
            batch_demos = "\n".join(demo_lines[batch_start:batch_end])
            batch_prompt = format_prompt(persona_prompt_template,
                persona_count=batch_count,
                domain=subject_context.get("domain", subject_context.get("category", "general")),
                subject=subject_context.get("query", ""),
                region=_region,
                reviewer_context=reviewer_ctx,
                demographics_list=batch_demos,
            )
        else:
            batch_prompt = persona_prompt

        response = await _llm.chat(
            model=gen_model,
            messages=[{"role": "user", "content": batch_prompt}],
            temperature=0.9,
            max_tokens=8192,
        )

        batch_personas = _extract_json_from_llm(response, expected_type="array")

        for i, persona in enumerate(batch_personas):
            global_idx = batch_start + i
            if global_idx < len(pre_sampled):
                persona["age"] = pre_sampled[global_idx].age
                persona["sex"] = pre_sampled[global_idx].sex
            persona["id"] = f"persona-{str(global_idx + 1).zfill(2)}"
            if not persona.get("region"):
                persona["region"] = _region

        all_personas.extend(batch_personas)

    return all_personas


def _save_personas_to_dir(personas: list[dict], personas_dir, region: str = "unknown"):
    """Save persona list as individual markdown files and a combined JSON."""
    from pathlib import Path
    personas_dir = Path(personas_dir)
    personas_dir.mkdir(parents=True, exist_ok=True)

    for persona in personas:
        pid = persona["id"]
        persona_md = f"# {pid}\n\n"
        persona_md += f"- **Name**: {persona.get('name', 'Anonymous')}\n"
        if persona.get("age"):
            persona_md += f"- **Age**: {persona['age']}\n"
        if persona.get("sex") and persona["sex"].lower() != "unspecified":
            persona_md += f"- **Sex**: {persona['sex']}\n"
        persona_md += f"- **Region**: {persona.get('region', region)}\n"
        persona_md += f"- **Background**: {persona.get('background', 'N/A')}\n"
        persona_md += f"- **Writing Tendencies**: {persona.get('writing_tendencies', 'N/A')}\n"
        priorities = persona.get("priorities", [])
        if isinstance(priorities, list):
            priorities = ", ".join(priorities)
        persona_md += f"- **Priorities**: {priorities}\n"

        persona_file = personas_dir / f"{pid}.md"
        persona_file.write_text(persona_md, encoding="utf-8")


async def _generate_writing_patterns(
    subject_context: dict,
    job_paths: dict,
    gen_model: str,
    api_key: str,
    usage_tracker=None,
) -> dict:
    """Generate domain-specific writing patterns via LLM.

    Returns the patterns dict (or empty dict on failure).
    """
    import json
    from pathlib import Path
    from cera.llm.openrouter import OpenRouterClient
    from cera.prompts import load_prompt, format_prompt

    _llm = OpenRouterClient(api_key=api_key, usage_tracker=usage_tracker, component="aml.patterns")
    _domain = subject_context.get("domain", subject_context.get("category", "general"))
    _region = subject_context.get("region", "unknown")

    # Build reference context from reference dataset (if available)
    # Check both datasets/ (new) and dataset/ (legacy) for reference files
    reference_context = ""
    _ref_candidates = [Path(job_paths["root"]) / "datasets", Path(job_paths["root"]) / "dataset"]
    dataset_dir = next((d for d in _ref_candidates if d.exists() and list(d.glob("reference_*"))), None)
    if dataset_dir:
        ref_files = list(dataset_dir.glob("reference_*"))
        if ref_files:
            try:
                with open(ref_files[0], "r") as f:
                    lines = f.readlines()[:20]
                sample_texts = []
                for line in lines:
                    try:
                        entry = json.loads(line.strip())
                        text = entry.get("text", entry.get("review_text", ""))
                        if text:
                            sample_texts.append(text[:200])
                    except json.JSONDecodeError:
                        continue
                if sample_texts:
                    reference_context = "## Reference Reviews (sample from real dataset)\n" + "\n".join(f"- \"{t}\"" for t in sample_texts[:10])
            except Exception:
                pass

    if not reference_context:
        reference_context = "No reference dataset available. Generate patterns based on the domain and subject."

    patterns_template = load_prompt("composition", "writing_patterns")
    patterns_prompt = format_prompt(patterns_template,
        domain=_domain,
        subject=subject_context.get("query", ""),
        region=_region,
        reference_context=reference_context,
    )

    response = await _llm.chat(
        model=gen_model,
        messages=[{"role": "user", "content": patterns_prompt}],
        temperature=0.7,
        max_tokens=4096,
    )

    return _extract_json_from_llm(response, expected_type="object")


async def _generate_structure_variants(
    subject_context: dict,
    reviewers_context: dict,
    estimated_reviews: int,
    gen_model: str,
    api_key: str,
    usage_tracker=None,
) -> list[dict]:
    """Generate review structure variants via LLM.

    Returns a list of structure variant dicts (or empty list on failure).
    """
    import math
    from cera.llm.openrouter import OpenRouterClient
    from cera.prompts import load_prompt, format_prompt

    _llm = OpenRouterClient(api_key=api_key, usage_tracker=usage_tracker, component="aml.structures")
    _domain = subject_context.get("domain", subject_context.get("category", "general"))
    _region = subject_context.get("region", "unknown")
    reviewer_ctx = reviewers_context.get("additional_context", "general consumer")

    variant_count = max(6, min(30, math.ceil(estimated_reviews * 0.15)))

    structure_template = load_prompt("composition", "structure_variants")
    structure_prompt = format_prompt(structure_template,
        variant_count=variant_count,
        domain=_domain,
        subject=subject_context.get("query", ""),
        region=_region,
        reviewer_context=reviewer_ctx,
    )

    response = await _llm.chat(
        model=gen_model,
        messages=[{"role": "user", "content": structure_prompt}],
        temperature=0.9,
        max_tokens=4096,
    )

    return _extract_json_from_llm(response, expected_type="array")


def build_persona_assignments(personas: list[dict], review_count: int) -> list[dict]:
    """Assign personas to reviews via shuffled round-robin (deck of cards pattern).

    Shuffles the full persona pool, deals sequentially. When exhausted,
    reshuffles and continues — guarantees even distribution while preventing
    predictable overlap at deck boundaries.
    """
    import random
    assignments = []
    deck = []
    for _ in range(review_count):
        if not deck:
            deck = list(personas)
            random.shuffle(deck)
        assignments.append(deck.pop())
    return assignments


def _extract_json_from_llm(text: str, expected_type: str = "array") -> any:
    """Extract and parse JSON from LLM response, handling common issues.

    Handles: markdown code fences, trailing commas, extra text around JSON.
    expected_type: "array" to find [...], "object" to find {...}
    """
    # Strip markdown code fences
    clean = text.strip()
    if clean.startswith("```"):
        clean = clean.split("\n", 1)[1] if "\n" in clean else clean[3:]
        if clean.endswith("```"):
            clean = clean[:-3]
        clean = clean.strip()

    # Try direct parse first
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        pass

    # Fix trailing commas before ] or } and retry
    fixed = re.sub(r',\s*([}\]])', r'\1', clean)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # Try to extract JSON structure from mixed text
    bracket = "[" if expected_type == "array" else "{"
    close = "]" if expected_type == "array" else "}"
    start = clean.find(bracket)
    if start != -1:
        # Find matching close bracket using depth tracking
        depth = 0
        in_string = False
        escape_next = False
        last_complete_item_end = -1  # Track end of last complete top-level item
        bracket_stack = []  # Track open brackets for repair
        for i in range(start, len(clean)):
            c = clean[i]
            if escape_next:
                escape_next = False
                continue
            if c == '\\' and in_string:
                escape_next = True
                continue
            if c == '"' and not escape_next:
                in_string = not in_string
                continue
            if in_string:
                continue
            if c in ('[', '{'):
                depth += 1
                bracket_stack.append(c)
            elif c in (']', '}'):
                depth -= 1
                if bracket_stack:
                    bracket_stack.pop()
                if depth == 0:
                    candidate = clean[start:i + 1]
                    # Try with trailing comma fix
                    candidate = re.sub(r',\s*([}\]])', r'\1', candidate)
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break
                # Track end of complete top-level array items (depth back to 1 = closed an item in the outer array)
                if depth == 1 and expected_type == "array":
                    last_complete_item_end = i

        # JSON was truncated (depth never reached 0) — try to salvage complete items
        if depth > 0 and expected_type == "array" and last_complete_item_end > start:
            # Cut at the last complete item and close the array
            truncated = clean[start:last_complete_item_end + 1]
            # Remove any trailing comma and close the array
            truncated = truncated.rstrip().rstrip(',') + ']'
            truncated = re.sub(r',\s*([}\]])', r'\1', truncated)
            try:
                result = json.loads(truncated)
                if isinstance(result, list) and len(result) > 0:
                    print(f"[JSON Repair] Recovered {len(result)} items from truncated response ({len(clean)} chars)")
                    return result
            except json.JSONDecodeError:
                pass

    raise json.JSONDecodeError("Could not extract valid JSON from LLM response", text, 0)


def format_persona_text(persona: dict) -> str:
    """Format a persona dict into the text block for the AML prompt."""
    parts = []
    if persona.get("name"):
        parts.append(f"- Name: {persona['name']}")
    if persona.get("age"):
        sex_str = f", {persona['sex'].capitalize()}" if persona.get("sex") and persona["sex"].lower() != "unspecified" else ""
        parts.append(f"- Age: {persona['age']}{sex_str}")
    elif persona.get("sex") and persona["sex"].lower() != "unspecified":
        parts.append(f"- Sex: {persona['sex'].capitalize()}")
    if persona.get("region"):
        parts.append(f"- Region: {persona['region']}")
    if persona.get("background"):
        parts.append(f"- Background: {persona['background']}")
    if persona.get("writing_tendencies"):
        parts.append(f"- Writing style: {persona['writing_tendencies']}")
    if persona.get("priorities"):
        priorities = persona["priorities"]
        if isinstance(priorities, list):
            priorities = ", ".join(priorities)
        parts.append(f"- Priorities: {priorities}")
    return "\n".join(parts) if parts else "- Background: general consumer"


def strip_url_citations(text: str) -> str:
    """Remove markdown URL citations like [source.com](https://...) from text."""
    import re
    return re.sub(r'\[([^\]]*)\]\(https?://[^\)]*\)', r'\1', text)


def assign_writing_patterns(patterns_data: dict) -> str:
    """Randomly pick ONE option per pattern category and format for AML prompt."""
    import random
    if not patterns_data or not patterns_data.get("patterns"):
        return ""
    lines = ["**Writing Patterns:**"]
    for _cat_key, cat_data in patterns_data["patterns"].items():
        context = cat_data.get("context", "")
        options = cat_data.get("options", [])
        if options:
            chosen = random.choice(options)
            lines.append(f'- {context}: write "{chosen}"')
    return "\n".join(lines) if len(lines) > 1 else ""


def format_structure_variant(variant: dict) -> str:
    """Format a structure variant dict into the text block for the AML prompt."""
    parts = []
    if variant.get("name"):
        parts.append(f"**Structure: {variant['name']}**")
    if variant.get("flow"):
        parts.append(f"- Flow: {variant['flow']}")
    if variant.get("sentiment_arc"):
        parts.append(f"- Sentiment arc: {variant['sentiment_arc']}")
    if variant.get("connectives"):
        parts.append(f"- Connectives: {variant['connectives']}")
    if variant.get("evidence"):
        parts.append(f"- Evidence style: {variant['evidence']}")
    if variant.get("rhythm"):
        parts.append(f"- Sentence rhythm: {variant['rhythm']}")
    return "\n".join(parts) if parts else ""


async def execute_generation(
    job_id: str,
    job_dir: str,
    config: JobConfig,
    api_key: str,
    convex_url: Optional[str],
    convex_token: Optional[str],
    should_complete_job: bool = True,  # Set to False when called from pipeline (evaluation will complete)
    model_tag: Optional[str] = None,  # Multi-model: tag for output filenames (e.g. "gemini-3-flash-preview")
    progress_callback=None,  # Multi-model: async callback(generated, failed, progress) instead of global convex update
    current_run: int = 1,  # Multi-run: which run this is (1-indexed)
    total_runs: int = 1,  # Multi-run: total number of runs
    dataset_dir_override: Optional[str] = None,  # Multi-target: override output directory
    amls_dir_override: Optional[str] = None,  # Multi-target: override AML prompts directory
    usage_tracker=None,  # Token usage tracker
) -> dict:
    """
    Execute the GENERATION phase of the CERA pipeline.

    This uses existing composition files (subject-context.json, etc.) from the job directory
    and generates reviews using the AML (Authenticity Modeling Layer).
    """
    import json
    import random
    import time
    from pathlib import Path
    from datetime import datetime

    from cera.pipeline.composition.rgm import ReviewerGenerationModule
    from cera.llm.openrouter import OpenRouterClient
    from cera.prompts import load_prompt, format_prompt

    convex = None
    if convex_url and convex_token:
        convex = ConvexClient(convex_url, convex_token, pocketbase_url=os.environ.get("POCKETBASE_URL"))
        print(f"[Generation] Convex client created for URL: {convex_url}")
    else:
        print(f"[Generation] WARNING: Convex client NOT created - url={convex_url}, token={'yes' if convex_token else 'no'}")

    job_path = Path(job_dir)
    contexts_path = job_path / "contexts"

    # Use directory overrides for multi-target support, otherwise derive from job_dir
    if dataset_dir_override:
        dataset_path = Path(dataset_dir_override)
    else:
        dataset_path = job_path / "datasets"

    if amls_dir_override:
        amls_path = Path(amls_dir_override)
    elif total_runs > 1:
        # Multi-run AML directory support: run subdirectories for multi-run, flat for single-run
        amls_path = job_path / "amls" / f"run-{current_run}"
    else:
        amls_path = job_path / "amls"

    # Ensure directories exist
    amls_path.mkdir(parents=True, exist_ok=True)
    dataset_path.mkdir(parents=True, exist_ok=True)

    # Track timing
    start_time = time.time()

    try:
        # ========================================
        # Load composition context files
        # ========================================
        print(f"[Generation] Loading context files from: {contexts_path}")

        with open(contexts_path / "subject-context.json") as f:
            subject_context = json.load(f)

        with open(contexts_path / "reviewers-context.json") as f:
            reviewers_context = json.load(f)

        with open(contexts_path / "attributes-context.json") as f:
            attributes_context = json.load(f)

        # Load verified facts (entity-clustered) if available
        verified_facts_path = contexts_path / "verified-facts.json"
        verified_facts = None
        verified_entities = None
        if verified_facts_path.exists():
            with open(verified_facts_path) as f:
                verified_facts = json.load(f)
            verified_entities = verified_facts.get("entities", [])
            print(f"[Generation] Loaded verified-facts.json with {len(verified_entities)} entities", flush=True)
        else:
            print(f"[Generation] No verified-facts.json found, using legacy feature subsampling", flush=True)

        # Load reference dataset sentences for style injection (if available)
        # Check both datasets/ (new) and dataset/ (legacy) for reference files
        reference_sentences: list[str] = []
        _ref_dirs = [job_path / "datasets", job_path / "dataset"]
        _ref_dir = next((d for d in _ref_dirs if d.exists() and list(d.glob("reference_*"))), None)
        if _ref_dir:
            ref_files = list(_ref_dir.glob("reference_*"))
            if ref_files:
                try:
                    _, ref_stats = load_texts_from_file(str(ref_files[0]), return_stats=True, return_sentences=True)
                    reference_sentences = ref_stats.get("sentence_texts", [])
                    if reference_sentences:
                        print(f"[Generation] Loaded {len(reference_sentences)} reference sentences for style injection")
                except Exception as e:
                    print(f"[Generation] Warning: Could not load reference sentences: {e}")

        # ========================================
        # Compute total_reviews early (needed for persona sizing)
        # ========================================
        count_mode = getattr(config.generation, 'count_mode', 'reviews')
        length_range = attributes_context["length_range"]
        avg_sentences_per_review = (length_range[0] + length_range[1]) / 2

        if count_mode == 'sentences':
            target_sentences = getattr(config.generation, 'target_sentences', None) or 1000
            estimated_reviews = int(target_sentences / avg_sentences_per_review)
            total_reviews = int(estimated_reviews * 1.3)  # 30% buffer for variance
        else:
            target_sentences = None
            total_reviews = config.generation.count

        # ========================================
        # Generate personas, patterns, and structure variants
        # (moved from Composition to Generation for multi-target support)
        # ========================================
        import math as _math

        # Check RGM ablation flag (controls PGM, WPM, SVM, and DEM)
        rgm_enabled = getattr(config.ablation, 'rgm_enabled', True) if config.ablation else True

        personas_pool: list[dict] = []
        writing_patterns: dict = {}
        structure_variants: list[dict] = []

        if rgm_enabled:
            if convex:
                await convex.add_log(job_id, "INFO", "Personas", "Starting persona generation for this target...")
                await convex.update_progress(job_id, 1, "Personas")

            # Determine persona count based on actual target size
            persona_ratio = getattr(config.reviewer_profile, 'persona_ratio', 0.9) or 0.9
            persona_count = max(1, _math.ceil(total_reviews * persona_ratio))

            # Check if personas already exist (from legacy composition or reuse)
            personas_json_path = contexts_path / "personas.json"
            if personas_json_path.exists():
                try:
                    with open(personas_json_path) as f:
                        existing_pool = json.load(f)
                    if len(existing_pool) >= persona_count:
                        personas_pool = existing_pool[:persona_count]
                        print(f"[Generation] Loaded {len(personas_pool)} personas from existing pool (sufficient for {persona_count} needed)")
                    else:
                        # Need more — generate the difference
                        print(f"[Generation] Existing pool has {len(existing_pool)} personas, need {persona_count}. Generating {persona_count - len(existing_pool)} more...")
                        try:
                            age_enabled = getattr(config.reviewer_profile, 'age_enabled', True) if hasattr(config.reviewer_profile, 'age_enabled') else True
                            sex_enabled = getattr(config.reviewer_profile, 'sex_enabled', True) if hasattr(config.reviewer_profile, 'sex_enabled') else True
                            extra = await _generate_personas(
                                persona_count=persona_count - len(existing_pool),
                                reviewer_profile=config.reviewer_profile,
                                reviewers_context=reviewers_context,
                                subject_context=subject_context,
                                gen_model=config.generation.model,
                                api_key=api_key,
                                age_enabled=age_enabled,
                                sex_enabled=sex_enabled,
                                usage_tracker=usage_tracker,
                            )
                            personas_pool = existing_pool + extra
                        except Exception as e:
                            print(f"[Generation] Warning: Could not generate additional personas: {e}")
                            personas_pool = existing_pool
                except Exception as e:
                    print(f"[Generation] Warning: Could not load existing personas: {e}")

            if not personas_pool:
                # Generate from scratch
                try:
                    age_enabled = getattr(config.reviewer_profile, 'age_enabled', True) if hasattr(config.reviewer_profile, 'age_enabled') else True
                    sex_enabled = getattr(config.reviewer_profile, 'sex_enabled', True) if hasattr(config.reviewer_profile, 'sex_enabled') else True
                    personas_pool = await _generate_personas(
                        persona_count=persona_count,
                        reviewer_profile=config.reviewer_profile,
                        reviewers_context=reviewers_context,
                        subject_context=subject_context,
                        gen_model=config.generation.model,
                        api_key=api_key,
                        age_enabled=age_enabled,
                        sex_enabled=sex_enabled,
                        usage_tracker=usage_tracker,
                    )
                    print(f"[Generation] Generated {len(personas_pool)} personas")

                    # Save to target-level personas dir (datasets/{target}/reviewer-personas/)
                    # dataset_dir_override is model dir: datasets/{target}/run{N}/model-slug/
                    # .parent.parent = datasets/{target}/ (target level)
                    if dataset_dir_override:
                        _target_dir = Path(dataset_dir_override).parent.parent
                    else:
                        _target_dir = Path(job_paths["root"])
                    _personas_save_dir = _target_dir / "reviewer-personas"
                    _save_personas_to_dir(personas_pool, _personas_save_dir, subject_context.get("region", "unknown"))

                    # Save JSON at target level and contexts level (for cross-target reuse)
                    _personas_json_save = _target_dir / "personas.json"
                    with open(_personas_json_save, "w") as f:
                        json.dump(personas_pool, f, indent=2)
                    # Also save to contexts for reuse by subsequent targets
                    _contexts_json_save = contexts_path / "personas.json"
                    with open(_contexts_json_save, "w") as f:
                        json.dump(personas_pool, f, indent=2)

                except Exception as e:
                    print(f"[Generation] Warning: Persona generation failed, will use RGM fallback: {e}")
                    if convex:
                        await convex.add_log(job_id, "WARN", "Personas", f"Persona generation failed ({e}), using RGM fallback")

            if convex and personas_pool:
                await convex.add_log(job_id, "INFO", "Personas", f"Personas ready: {len(personas_pool)} for {total_reviews} reviews")
                await convex.update_progress(job_id, 2, "Personas")

            # Generate writing patterns (or load from existing)
            patterns_path = contexts_path / "writing-patterns.json"
            if patterns_path.exists():
                try:
                    with open(patterns_path) as f:
                        writing_patterns = json.load(f)
                    print(f"[Generation] Loaded {len(writing_patterns.get('patterns', {}))} writing pattern categories from existing")
                except Exception as e:
                    print(f"[Generation] Warning: Could not load writing patterns: {e}")

            if not writing_patterns:
                try:
                    writing_patterns = await _generate_writing_patterns(
                        subject_context=subject_context,
                        job_paths=job_paths,
                        gen_model=config.generation.model,
                        api_key=api_key,
                        usage_tracker=usage_tracker,
                    )
                    # Save for reuse
                    _patterns_save = contexts_path / "writing-patterns.json"
                    with open(_patterns_save, "w") as f:
                        json.dump(writing_patterns, f, indent=2)
                    print(f"[Generation] Generated {len(writing_patterns.get('patterns', {}))} writing pattern categories")
                except Exception as e:
                    print(f"[Generation] Warning: Pattern generation failed: {e}")

            # Generate structure variants (per-target: variant count scales with target size)
            import math as _sv_math
            _needed_variants = max(6, min(30, _sv_math.ceil(total_reviews * 0.15)))

            # Check target-level first, then contexts-level fallback
            if dataset_dir_override:
                _target_structs_path = Path(dataset_dir_override).parent.parent / "structure-variants.json"
            else:
                _target_structs_path = None
            _contexts_structs_path = contexts_path / "structure-variants.json"

            for _sp in [_target_structs_path, _contexts_structs_path]:
                if _sp and _sp.exists():
                    try:
                        with open(_sp) as f:
                            _loaded = json.load(f)
                        if len(_loaded) >= _needed_variants:
                            structure_variants = _loaded[:_needed_variants]
                            print(f"[Generation] Loaded {len(structure_variants)} structure variants from {_sp.name} (needed {_needed_variants})")
                            break
                        else:
                            print(f"[Generation] Found {len(_loaded)} variants in {_sp.name}, need {_needed_variants} — regenerating")
                    except Exception as e:
                        print(f"[Generation] Warning: Could not load structure variants from {_sp.name}: {e}")

            if not structure_variants:
                try:
                    structure_variants = await _generate_structure_variants(
                        subject_context=subject_context,
                        reviewers_context=reviewers_context,
                        estimated_reviews=total_reviews,
                        gen_model=config.generation.model,
                        api_key=api_key,
                        usage_tracker=usage_tracker,
                    )
                    # Save at target level
                    if _target_structs_path:
                        _target_structs_path.parent.mkdir(parents=True, exist_ok=True)
                        with open(_target_structs_path, "w") as f:
                            json.dump(structure_variants, f, indent=2)
                    # Also save to contexts for fallback
                    with open(_contexts_structs_path, "w") as f:
                        json.dump(structure_variants, f, indent=2)
                    print(f"[Generation] Generated {len(structure_variants)} structure variants for {total_reviews} reviews")
                except Exception as e:
                    print(f"[Generation] Warning: Structure generation failed: {e}")
        else:
            # RGM disabled — skip PGM, WPM, SVM
            print("[Generation] RGM disabled — skipping persona generation, writing patterns, and structure variants")
            if convex:
                await convex.add_log(job_id, "INFO", "RGM", "RGM disabled — skipping personas, writing patterns, structure variants, and diversity enforcement")
                await convex.update_progress(job_id, 2, "RGM")

        if convex:
            await convex.add_log(job_id, "INFO", "AML", "Starting generation phase...")
            await convex.add_log(job_id, "INFO", "AML", f"Subject: {subject_context.get('query', 'N/A')}")
            if reference_sentences:
                await convex.add_log(job_id, "INFO", "AML", f"Reference style injection: {len(reference_sentences)} sentences loaded")
            if personas_pool:
                await convex.add_log(job_id, "INFO", "AML", f"Personas: {len(personas_pool)} ready")
            if writing_patterns:
                await convex.add_log(job_id, "INFO", "AML", f"Writing patterns: {len(writing_patterns.get('patterns', {}))} categories")
            if structure_variants:
                await convex.add_log(job_id, "INFO", "AML", f"Structure variants: {len(structure_variants)} templates")
            await convex.update_progress(job_id, 5, "AML")

        # ========================================
        # Initialize components
        # ========================================

        # RGM for generating reviewer profiles
        # age_range can be None if age ablation is disabled
        age_range = reviewers_context.get("age_range")
        rgm = ReviewerGenerationModule(
            age_range=tuple(age_range) if age_range else None,
            sex_distribution=reviewers_context["sex_distribution"],
            additional_context=reviewers_context.get("additional_context"),
        )

        # Generation settings (count_mode, length_range, avg_sentences_per_review,
        # target_sentences, estimated_reviews, total_reviews already computed above)
        if count_mode == 'sentences':
            print(f"[Generation] Mode: Sentences | Target: {target_sentences} sentences")
            print(f"[Generation] Estimated ~{estimated_reviews} reviews (generating up to {total_reviews} with buffer)")
        else:
            print(f"[Generation] Mode: Reviews | Target: {total_reviews} reviews")

        batch_size = config.generation.batch_size
        request_size = config.generation.request_size

        # Pre-assign personas to reviews via shuffled round-robin
        persona_assignments: list[dict] = []
        if personas_pool:
            persona_assignments = build_persona_assignments(personas_pool, total_reviews)
            print(f"[Generation] Pre-assigned {len(persona_assignments)} persona slots from pool of {len(personas_pool)}")

        # Pre-assign structure variants to reviews via shuffled round-robin
        structure_assignments: list[dict] = []
        if structure_variants:
            structure_assignments = build_persona_assignments(structure_variants, total_reviews)
            print(f"[Generation] Pre-assigned {len(structure_assignments)} structure variant slots from pool of {len(structure_variants)}")

        # Build model ID - check if model already has provider prefix
        model_name = config.generation.model
        provider = config.generation.provider
        if "/" in model_name:
            # Model already has provider prefix (e.g., "mistralai/mistral-7b")
            model = model_name
        else:
            # Add provider prefix
            model = f"{provider}/{model_name}"

        # Detect local vLLM model (prefixed with "local/")
        is_local_model, actual_model = _parse_local_model(model)
        if is_local_model:
            local_endpoint, local_api_key = await _get_local_llm_settings(convex)
            print(f"[Generation] Using LOCAL vLLM model: {actual_model} at {local_endpoint}")
            model = actual_model  # Strip local/ prefix for vLLM API
        else:
            print(f"[Generation] Using model: {model} (provider={provider}, model_name={model_name})")

        # LLM client for generation calls
        _target_label = str(config.generation.count)
        _llm_kwargs = dict(
            usage_tracker=usage_tracker, component="aml",
            target=_target_label, run=f"run{current_run}" if total_runs > 1 else "",
        )
        if is_local_model:
            llm = OpenRouterClient(api_key=local_api_key, base_url=local_endpoint, **_llm_kwargs)
        else:
            llm = OpenRouterClient(api_key=api_key, **_llm_kwargs)

        # Polarity distribution (length_range already set above for sentence count calculation)
        polarity_dist = attributes_context["polarity"]
        temp_range = config.attributes_profile.temp_range

        if convex:
            # Log detailed configuration
            await convex.add_log(job_id, "INFO", "Config", f"Model: {model}")
            if count_mode == 'sentences':
                await convex.add_log(job_id, "INFO", "Config", f"Mode: Sentences | Target: {target_sentences} sentences (~{estimated_reviews} reviews)")
            else:
                await convex.add_log(job_id, "INFO", "Config", f"Mode: Reviews | Target: {total_reviews} reviews")
            await convex.add_log(
                job_id, "INFO", "Config",
                f"Polarity: {int(polarity_dist['positive']*100)}% pos, {int(polarity_dist['neutral']*100)}% neu, {int(polarity_dist['negative']*100)}% neg"
            )
            await convex.add_log(job_id, "INFO", "Config", f"Length: {length_range[0]}-{length_range[1]} sentences, Temp: {temp_range[0]}-{temp_range[1]}")
            await convex.add_log(job_id, "INFO", "AML", "Configuration loaded, starting generation...")

        # ========================================
        # Load prompt templates
        # ========================================
        try:
            system_template = load_prompt("aml", "system")
            user_template = load_prompt("aml", "user")
        except FileNotFoundError:
            # Fallback to inline templates if files not found
            system_template = """You are an authentic review writer creating realistic {domain} reviews.

Guidelines:
- Write naturally and authentically
- Include specific details when appropriate
- Match the sentiment (positive/neutral/negative) as instructed
- Keep the review {min_sentences} to {max_sentences} sentences long
- Write from the perspective of a {age}-year-old {sex}"""

            user_template = """Write a {polarity} review for: {subject}

Reviewer: {age}-year-old {sex} from {region}
Length: {min_sentences}-{max_sentences} sentences

Aspect Categories: {aspect_categories}

{dataset_mode_instruction}

Output as JSON:
{output_example}

Output ONLY the JSON object, no other text."""

        # ========================================
        # Generate reviews
        # ========================================
        reviews = []
        generated_count = 0
        total_sentences_generated = 0  # Track sentences for sentence-based stopping

        # Calculate digits for padding AML file names
        digits = max(1, len(str(total_reviews)))

        # Note: Polarity distribution is now applied at SENTENCE level, not review level
        # Each review will contain a mix of positive, neutral, and negative sentences
        # based on the polarity_dist configuration

        print(f"[Generation] Starting generation of {total_reviews} reviews")
        print(f"[Generation] Sentence-level polarity distribution: {int(polarity_dist['positive']*100)}% pos, {int(polarity_dist['neutral']*100)}% neu, {int(polarity_dist['negative']*100)}% neg")

        # Track errors for logging
        error_count = 0
        malformed_count = 0  # Track JSON validation failures
        last_error_msg = ""
        last_progress_update = 0

        # Rate limiting — local vLLM models skip OpenRouter tier checks entirely
        if is_local_model:
            REQUEST_DELAY = 0.05  # Minimal delay for local inference
            print(f"[Generation] Local vLLM model - no rate limiting: {REQUEST_DELAY}s between requests")
            if convex:
                await convex.add_log(job_id, "INFO", "AML", "Local vLLM model - no rate limiting")
        else:
            # Check if using a free model (ends with :free) - these ALWAYS have rate limits
            is_free_model = model.endswith(":free")

            # Check OpenRouter account tier
            tier_info = await check_openrouter_tier(api_key)
            is_free_account = tier_info["is_free_tier"]

            # Rate limiting: Apply for free models OR free accounts
            # Free models have rate limits regardless of account tier
            needs_rate_limiting = is_free_model or is_free_account

            if needs_rate_limiting:
                REQUEST_DELAY = 3.5  # seconds between requests (~17 req/min, under 20 req/min limit)
                reason = "free model" if is_free_model else "free tier account"
                print(f"[Generation] Rate limiting enabled ({reason}): {REQUEST_DELAY}s between requests")
                if convex:
                    await convex.add_log(job_id, "INFO", "AML", f"Rate limiting ({reason}): {REQUEST_DELAY}s between requests (~{60/REQUEST_DELAY:.0f} req/min)")
            else:
                REQUEST_DELAY = 0.1  # Minimal delay for paid tier with paid models
                print(f"[Generation] Paid tier + paid model - fast generation enabled: {REQUEST_DELAY}s")
                if convex:
                    await convex.add_log(job_id, "INFO", "AML", "Paid tier + paid model - fast generation enabled")

        MAX_RETRIES = 3
        MAX_VALIDATION_RETRIES = 2  # Additional retries specifically for malformed JSON
        last_request_time = 0

        def validate_review_json(text: str, dataset_mode: str) -> tuple[dict | None, str | None]:
            """
            Validate and parse LLM JSON response.

            Returns:
                (parsed_dict, None) on success
                (None, error_message) on failure
            """
            if not text or not isinstance(text, str):
                return None, "Empty or invalid response"

            clean_text = text.strip()

            # Strip markdown code fences if present
            if clean_text.startswith("```"):
                try:
                    first_newline = clean_text.index("\n")
                    clean_text = clean_text[first_newline + 1:]
                except ValueError:
                    return None, "Malformed markdown fence (no newline after opening)"
            if clean_text.rstrip().endswith("```"):
                clean_text = clean_text.rstrip()[:-3].rstrip()

            # Try to parse JSON
            try:
                parsed = json.loads(clean_text)
            except json.JSONDecodeError as e:
                # Try to extract JSON from response if it contains extra text
                import re
                json_match = re.search(r'\{[\s\S]*\}', clean_text)
                if json_match:
                    try:
                        parsed = json.loads(json_match.group())
                    except json.JSONDecodeError:
                        return None, f"JSON parse error: {str(e)[:100]}"
                else:
                    return None, f"JSON parse error: {str(e)[:100]}"

            # Validate schema: must have "sentences" array
            if not isinstance(parsed, dict):
                return None, "Response is not a JSON object"

            if "sentences" not in parsed:
                return None, "Missing 'sentences' array in response"

            sentences = parsed["sentences"]
            if not isinstance(sentences, list):
                return None, "'sentences' is not an array"

            if len(sentences) == 0:
                return None, "'sentences' array is empty"

            # Validate each sentence structure
            for idx, sent in enumerate(sentences):
                if not isinstance(sent, dict):
                    return None, f"Sentence {idx+1} is not an object"

                if "text" not in sent:
                    return None, f"Sentence {idx+1} missing 'text' field"

                if not isinstance(sent.get("text", ""), str) or not sent["text"].strip():
                    return None, f"Sentence {idx+1} has empty or invalid 'text'"

                # Validate opinions array (can be empty for sentences with no aspect mentions)
                opinions = sent.get("opinions", [])
                if not isinstance(opinions, list):
                    return None, f"Sentence {idx+1} 'opinions' is not an array"

                for op_idx, opinion in enumerate(opinions):
                    if not isinstance(opinion, dict):
                        return None, f"Sentence {idx+1}, opinion {op_idx+1} is not an object"

                    # Required fields for all modes
                    if "category" not in opinion:
                        return None, f"Sentence {idx+1}, opinion {op_idx+1} missing 'category'"
                    if "polarity" not in opinion:
                        return None, f"Sentence {idx+1}, opinion {op_idx+1} missing 'polarity'"

                    # Validate polarity value
                    polarity_val = opinion.get("polarity", "").lower()
                    if polarity_val not in ["positive", "negative", "neutral"]:
                        return None, f"Sentence {idx+1}, opinion {op_idx+1} invalid polarity: '{polarity_val}'"

                    # For explicit mode, target is required (can be "NULL" for implicit aspects)
                    if dataset_mode == "explicit":
                        if "target" not in opinion:
                            return None, f"Sentence {idx+1}, opinion {op_idx+1} missing 'target' (explicit mode)"

            return parsed, None

        # ========================================
        # Parallel Generation with request_size concurrency
        # ========================================
        semaphore = asyncio.Semaphore(request_size)
        rate_limit_lock = asyncio.Lock()  # For coordinating rate limiting across parallel requests
        last_request_times = [0.0]  # Mutable container for shared state

        # Keep full feature pools for per-review subsampling (diversity)
        all_features = subject_context.get("characteristics", [])
        all_pros = subject_context.get("positives", [])
        all_cons = subject_context.get("negatives", [])
        # When pros/cons are empty (common with SIL output), derive synthetic pools from characteristics
        if not all_pros and all_features:
            all_pros = [f.split('.')[0].strip() for f in all_features[:5]]
        if not all_cons:
            all_cons = ["minor issues", "occasional inconsistencies", "room for improvement", "could be better in some areas"]
        _domain = subject_context.get("domain", subject_context.get("category", "N/A"))
        _region = subject_context.get("region", "N/A")
        dataset_mode = getattr(config.generation, "dataset_mode", "explicit")

        # Get aspect categories - try request config first, then job's config.json, then defaults
        aspect_cats = getattr(config.subject_profile, "aspect_categories", None) if config.subject_profile else None
        if not aspect_cats or len(aspect_cats) == 0:
            # Try to read from job's config.json (for GEN+EVAL only mode where composition was reused)
            job_config_path = job_path / "config.json"
            if job_config_path.exists():
                try:
                    with open(job_config_path) as f:
                        job_config_data = json.load(f)
                    aspect_cats = job_config_data.get("subject_profile", {}).get("aspect_categories", [])
                    if aspect_cats:
                        print(f"[Generation] Loaded {len(aspect_cats)} aspect categories from job config.json", flush=True)
                except Exception as e:
                    print(f"[Generation] Warning: Could not read aspect categories from config.json: {e}", flush=True)
        if not aspect_cats or len(aspect_cats) == 0:
            aspect_cats = ["PRODUCT#QUALITY", "PRODUCT#PRICES", "SERVICE#GENERAL", "EXPERIENCE#GENERAL"]
            print(f"[Generation] Using default aspect categories", flush=True)

        # Initialize diversity enforcement modules
        from cera.pipeline.generation.neb import NegativeExampleBuffer
        from cera.pipeline.generation.vocab_tracker import VocabDiversityTracker
        # Pass subject features + pros/cons as exempt terms so product vocabulary isn't penalized
        _subject_terms = all_features + all_pros + all_cons
        vocab_tracker = VocabDiversityTracker(top_k=10, subject_terms=_subject_terms)
        neb_buffer = None

        if rgm_enabled:
            # DEM enabled — initialize NEB, opening directives, cap styles
            neb_enabled = getattr(config.generation, 'neb_enabled', True)
            neb_depth = getattr(config.generation, 'neb_depth', 3)
            if neb_enabled and neb_depth > 0:
                max_buffer_size = neb_depth * request_size
                neb_buffer = NegativeExampleBuffer(max_size=max_buffer_size)
                print(f"[NEB] Enabled with depth={neb_depth}, max_buffer={max_buffer_size} reviews", flush=True)
                if convex:
                    await convex.add_log(job_id, "INFO", "NEB", f"Negative Example Buffer enabled (depth={neb_depth}, max={max_buffer_size})")
            else:
                print(f"[NEB] Disabled (neb_enabled={neb_enabled}, neb_depth={neb_depth})", flush=True)

            # Opening directives for per-review diversity enforcement
            OPENING_DIRECTIVES = [
                "Start with a specific product detail, measurement, or physical observation you noticed immediately",
                "Start mid-thought or mid-story, as if continuing a conversation (e.g., 'So I finally...' or 'Three weeks in and...')",
                "Start with a rhetorical question or genuine question to the reader",
                "Start with a raw emotional reaction -- excitement, disappointment, surprise, or frustration",
                "Start with when, where, or how you acquired/visited/discovered this (time and place context)",
                "Start with a comparison to a competitor, alternative, or your previous experience",
                "Start with a casual filler word or interjection (e.g., 'Ok so...', 'Man,', 'Alright,', 'Look,')",
                "Start with a direct complaint or frustration about a specific issue",
                "Start with enthusiastic praise or a strong recommendation",
                "Start with a warning, caveat, or 'heads up' to other buyers/visitors",
                "Start by addressing the reader directly (e.g., 'If you're looking for...', 'For anyone considering...')",
                "Start with a time reference (e.g., 'After two months...', 'First day with this...')",
                "Start with a contradictory or nuanced take (e.g., 'I wanted to love this but...', 'Mixed feelings...')",
                "Start with a factual statement about usage (e.g., 'Used this daily for 3 months...')",
                "Start with a story or anecdote about why you bought/visited this",
            ]

            # Capitalization styles for per-review authenticity (weighted distribution)
            # Read weights from config or use defaults (decimal values, e.g., 0.55 = 55%)
            cap_weights = getattr(config.attributes_profile, 'cap_weights', None) or {}
            _cap_w = {
                "standard": cap_weights.get("standard", 0.55),
                "lowercase": cap_weights.get("lowercase", 0.20),
                "mixed": cap_weights.get("mixed", 0.15),
                "emphasis": cap_weights.get("emphasis", 0.10),
            }
            CAPITALIZATION_STYLES = [
                ("Write with standard/proper capitalization. Capitalize the first word of each sentence normally.", _cap_w["standard"]),
                ("Write in mostly lowercase. Start your review and most sentences with lowercase letters. Use lowercase 'i' instead of 'I'. Example: 'honestly this laptop is pretty solid. i use it every day and the keyboard feels great.'", _cap_w["lowercase"]),
                ("Write with casual/mixed capitalization. Start the review lowercase, occasionally skip capitalization after periods, but not consistently. Example: 'so i picked this up last week. The screen is decent but the speakers kinda suck.'", _cap_w["mixed"]),
                ("Write with occasional ALL CAPS for emphasis on key words, but otherwise normal capitalization. Example: 'The battery life is AMAZING but the trackpad is SO frustrating to use.'", _cap_w["emphasis"]),
            ]
        else:
            # RGM disabled — DEM off: no NEB, no opening directives, standard cap only
            print("[Generation] RGM disabled — DEM off (no NEB, no vocab tracking, no opening directives, standard capitalization)", flush=True)
            OPENING_DIRECTIVES = [""]
            CAPITALIZATION_STYLES = [
                ("Write with standard/proper capitalization. Capitalize the first word of each sentence normally.", 1.0),
            ]

        if dataset_mode == "implicit":
            dataset_mode_instruction = "For implicit mode: do NOT include target terms. Only provide the category and polarity for each opinion."
            output_example = '{\n  "sentences": [\n    {\n      "text": "The food was absolutely divine.",\n      "opinions": [\n        {"category": "FOOD#QUALITY", "polarity": "positive"}\n      ]\n    }\n  ]\n}'
        else:
            dataset_mode_instruction = "For explicit mode: include the target term from the sentence text for each opinion. IMPORTANT: The target MUST be a SINGLE WORD (noun) in lowercase that appears exactly in the text."
            output_example = '{\n  "sentences": [\n    {\n      "text": "The carbonara was absolutely divine.",\n      "opinions": [\n        {"target": "carbonara", "category": "FOOD#QUALITY", "polarity": "positive"}\n      ]\n    }\n  ]\n}'

        async def generate_single_review(review_index: int, neb_buffer_param, opening_directive: str = "", capitalization_style: str = "") -> dict:
            """Generate a single review with retries. Returns result dict."""
            # Use explicitly passed neb_buffer to avoid closure capture issues
            neb_buffer = neb_buffer_param
            async with semaphore:
                # Edge length sampling: range-based probability distribution
                # minEdge range: [min_length, normal_floor - 1] (floor inclusive, ceiling exclusive of normal)
                # normal range:  [normal_floor, normal_ceiling] (both inclusive)
                # maxEdge range: [normal_ceiling + 1, max_length] (floor exclusive of normal, ceiling inclusive)
                edge_cfg = attributes_context.get("edge_lengths")
                if edge_cfg:
                    roll = random.random()
                    min_chance = edge_cfg.get("min_chance", 0)
                    max_chance = edge_cfg.get("max_chance", 0)
                    min_length = edge_cfg.get("min_length", 1)
                    max_length = edge_cfg.get("max_length", 15)
                    # Build edge ranges relative to normal length_range
                    min_edge_floor = min_length
                    min_edge_ceil = length_range[0] - 1  # exclusive of normal floor
                    max_edge_floor = length_range[1] + 1  # exclusive of normal ceiling
                    max_edge_ceil = max_length
                    if roll < min_chance and min_edge_floor <= min_edge_ceil:
                        num_sentences = random.randint(min_edge_floor, min_edge_ceil)
                    elif roll < min_chance + max_chance and max_edge_floor <= max_edge_ceil:
                        num_sentences = random.randint(max_edge_floor, max_edge_ceil)
                    else:
                        num_sentences = random.randint(length_range[0], length_range[1])
                else:
                    num_sentences = random.randint(length_range[0], length_range[1])
                # Sample temperature in 0.1 increments (e.g., 0.7, 0.8, 0.9)
                temp_steps = [round(temp_range[0] + i * 0.1, 1)
                              for i in range(int((temp_range[1] - temp_range[0]) / 0.1) + 1)]
                temperature = random.choice(temp_steps)

                # Get NEB context (empty string if disabled or first batch)
                neb_context = ""
                if neb_buffer and len(neb_buffer) > 0:
                    neb_context = neb_buffer.get_formatted_context()
                elif not neb_buffer:
                    pass  # NEB disabled

                # Get vocabulary diversity context (cumulative phrase tracking)
                vocab_context = vocab_tracker.get_formatted_context()

                # Reference style injection (sample 2-3 real review sentences as phrasing examples)
                style_examples = ""
                if reference_sentences and len(reference_sentences) >= 3:
                    k_style = random.randint(2, 3)
                    sampled_refs = random.sample(reference_sentences, min(k_style, len(reference_sentences)))
                    examples_list = "\n".join(f'- "{s.strip()}"' for s in sampled_refs if s.strip())
                    if examples_list:
                        style_examples = (
                            "## Phrasing Style Reference\n"
                            "The following are excerpts from real reviews. Write in a similar natural, human tone.\n"
                            "Do NOT copy their content -- only mimic their phrasing style, word choices, and tone:\n\n"
                            f"{examples_list}\n\n"
                            "Match this level of informality, sentence structure variety, and vocabulary."
                        )

                # Per-review entity selection (or legacy feature subsampling)
                if verified_entities:
                    # Pick one entity for this review — ALL its facts, no subsampling
                    entity = random.choice(verified_entities)
                    review_features = entity.get("characteristics", [])
                    review_pros = entity.get("positives", [])
                    review_cons = entity.get("negatives", [])
                    # Fallback: if entity has empty characteristics, use flat pools from subject_context
                    if not review_features:
                        review_features = list(all_features) if all_features else []
                    if not review_pros:
                        review_pros = list(all_pros) if all_pros else []
                    if not review_cons:
                        review_cons = list(all_cons) if all_cons else []
                    # Use only this entity's applicable aspects for the sentence plan
                    entity_aspects = entity.get("applicable_aspects", aspect_cats)
                    if not entity_aspects:
                        entity_aspects = aspect_cats
                    # Derive pros/cons fallbacks from entity characteristics
                    if not review_pros and review_features:
                        review_pros = [f.split('.')[0].strip() for f in review_features[:3]]
                    if not review_cons:
                        review_cons = ["minor issues", "could be better in some areas"]
                else:
                    # Legacy: random subsampling from flat pools
                    entity_aspects = aspect_cats
                    if all_features:
                        core = all_features[:2]
                        extras = all_features[2:]
                        k_extra = min(random.randint(1, 2), len(extras)) if extras else 0
                        review_features = core + random.sample(extras, k_extra) if k_extra > 0 else list(core)
                    else:
                        review_features = []
                    k_pros = min(random.randint(1, 2), len(all_pros)) if all_pros else 0
                    k_cons = min(random.randint(1, 2), len(all_cons)) if all_cons else 0
                    review_pros = random.sample(all_pros, k_pros) if k_pros > 0 else []
                    review_cons = random.sample(all_cons, k_cons) if k_cons > 0 else []

                # Build aspect-based sentence plan (uses entity-scoped or full aspect pool)
                sentence_plan = build_sentence_plan(num_sentences, entity_aspects, polarity_dist)
                aspect_sentence_plan_str = format_sentence_plan(sentence_plan)

                # Get persona for this review (from pre-assigned shuffled round-robin pool)
                reviewer_age = None
                reviewer_sex = "unspecified"
                if persona_assignments and review_index < len(persona_assignments):
                    persona = persona_assignments[review_index]
                    persona_text = format_persona_text(persona)
                    reviewer_age = persona.get("age")
                    reviewer_sex = persona.get("sex", "unspecified")
                else:
                    # Fallback: use RGM-generated profile with shared background
                    reviewer = rgm.generate_profile()
                    reviewer_age = reviewer.age
                    reviewer_sex = reviewer.sex
                    fallback_parts = []
                    if reviewer.age is not None:
                        fallback_parts.append(f"- Age: {reviewer.age}")
                    if reviewer.sex and reviewer.sex.lower() != "unspecified":
                        fallback_parts.append(f"- Sex: {reviewer.sex}")
                    region = subject_context.get("region", "")
                    if region:
                        fallback_parts.append(f"- Region: {region}")
                    additional_ctx = reviewers_context.get("additional_context", "general consumer")
                    if additional_ctx:
                        fallback_parts.append(f"- Background: {additional_ctx}")
                    persona_text = "\n".join(fallback_parts) if fallback_parts else "- Background: general consumer"

                # Assign writing patterns for this review
                writing_pattern_assignments_str = assign_writing_patterns(writing_patterns)

                # Assign structure variant for this review
                structure_variant_str = ""
                if structure_assignments and review_index < len(structure_assignments):
                    structure_variant_str = format_structure_variant(structure_assignments[review_index])

                # Strip URLs from SIL features for clean Subject Intelligence block
                features_clean = strip_url_citations(", ".join(review_features)) if review_features else "N/A"

                prompt_vars = {
                    "subject": subject_context["query"],
                    "domain": _domain,
                    "region": subject_context.get("region", ""),
                    "features_no_urls": features_clean,
                    "persona_text": persona_text,
                    "pros": strip_url_citations(", ".join(review_pros)) if review_pros else "quality and value",
                    "cons": strip_url_citations(", ".join(review_cons)) if review_cons else "minor issues",
                    "num_sentences": num_sentences,
                    "aspect_sentence_plan": aspect_sentence_plan_str,
                    "dataset_mode_instruction": dataset_mode_instruction,
                    "output_example": output_example,
                    "neb_context": neb_context,
                    "vocab_diversity": vocab_context,
                    "style_examples": style_examples,
                    "opening_directive": opening_directive,
                    "capitalization_style": capitalization_style,
                    "writing_pattern_assignments": writing_pattern_assignments_str,
                    "structure_variant": structure_variant_str,
                }

                system_prompt = format_prompt(system_template, **prompt_vars)
                user_prompt = format_prompt(user_template, **prompt_vars)

                aml_number = str(review_index).zfill(digits)
                aml_file = amls_path / f"aml-{aml_number}.md"
                aml_content = f"# AML Prompt {aml_number}\n\n## System Prompt\n{system_prompt}\n\n## User Prompt\n{user_prompt}\n"
                aml_file.write_text(aml_content, encoding="utf-8")

                review_text = None
                parsed_review = None
                validation_error = None
                validation_retries = 0
                local_error = False
                local_malformed = False
                local_error_msg = None

                for retry in range(MAX_RETRIES + MAX_VALIDATION_RETRIES):
                    try:
                        async with rate_limit_lock:
                            elapsed_since_last = time.time() - last_request_times[0]
                            if elapsed_since_last < REQUEST_DELAY:
                                await asyncio.sleep(REQUEST_DELAY - elapsed_since_last)
                            last_request_times[0] = time.time()

                        messages = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ]

                        if validation_error and validation_retries > 0:
                            correction_prompt = f"""Your previous response was invalid: {validation_error}

Please respond with ONLY a valid JSON object (no markdown fences, no extra text).
The JSON must have this exact structure:
{{
  "sentences": [
    {{
      "text": "The sentence text here.",
      "opinions": [
        {{"target": "aspect term", "category": "CATEGORY#ATTRIBUTE", "polarity": "positive|negative|neutral"}}
      ]
    }}
  ]
}}

Requirements:
- Every sentence must have "text" (non-empty string)
- Every opinion must have "category" and "polarity"
- Polarity must be exactly "positive", "negative", or "neutral"
{"- Every opinion must have 'target' (use 'NULL' for implicit aspects)" if dataset_mode == "explicit" else ""}
"""
                            messages.append({"role": "assistant", "content": review_text})
                            messages.append({"role": "user", "content": correction_prompt})

                        review_text = await llm.chat(
                            model=model,
                            messages=messages,
                            temperature=temperature if validation_retries == 0 else max(0.3, temperature - 0.2),
                            max_tokens=4096,
                        )

                        parsed_review, validation_error = validate_review_json(review_text, dataset_mode)

                        if parsed_review:
                            break
                        else:
                            validation_retries += 1
                            if validation_retries <= MAX_VALIDATION_RETRIES:
                                print(f"[Generation] Review {review_index+1} validation failed (retry {validation_retries}): {validation_error}")
                                continue
                            else:
                                local_malformed = True
                                print(f"[Generation] Review {review_index+1} abandoned after {validation_retries} retries: {validation_error}")
                                break

                    except Exception as e:
                        error_msg = str(e)
                        is_rate_limit = "429" in error_msg or "rate" in error_msg.lower()

                        if is_rate_limit and retry < MAX_RETRIES - 1:
                            backoff_time = (2 ** retry) * 5
                            print(f"[Generation] Rate limited on review {review_index+1}, waiting {backoff_time}s")
                            await asyncio.sleep(backoff_time)
                            continue
                        else:
                            local_error = True
                            local_error_msg = error_msg
                            print(f"[Generation] Error generating review {review_index+1}: {error_msg}")
                            break

                return {
                    "index": review_index,
                    "aml_number": aml_number,
                    "error": local_error,
                    "malformed": local_malformed,
                    "error_msg": local_error_msg,
                    "reviewer_age": reviewer_age,
                    "reviewer_sex": reviewer_sex,
                    "num_sentences": num_sentences,
                    "temperature": temperature,
                    "parsed_review": parsed_review,
                    "review_text": review_text,
                }

        # Process in batches for progress tracking
        print(f"[Generation] Starting parallel generation with {request_size} concurrent requests")
        if convex:
            await convex.add_log(job_id, "INFO", "AML", f"Parallel generation: {request_size} concurrent requests")

        batch_start = 0
        stop_generation = False

        while batch_start < total_reviews and not stop_generation:
            if convex and batch_start > 0:
                job_status = await convex.get_job(job_id)
                if job_status and job_status.get("status") in ["terminated", "paused"]:
                    status = job_status.get("status")
                    print(f"[Generation] Job {status} by user")
                    await convex.update_progress(job_id, 100, status.capitalize())
                    return {"status": status, "jobId": job_id, "generatedCount": generated_count}

            batch_end = min(batch_start + request_size, total_reviews)
            print(f"[NEB DEBUG] At batch creation: neb_buffer is {'not None (id=' + str(id(neb_buffer)) + ')' if neb_buffer else 'None'}", flush=True)
            # Sample unique opening directives for this batch
            batch_size = batch_end - batch_start
            if batch_size <= len(OPENING_DIRECTIVES):
                batch_directives = random.sample(OPENING_DIRECTIVES, batch_size)
            else:
                batch_directives = OPENING_DIRECTIVES.copy()
                extras = random.choices(OPENING_DIRECTIVES, k=batch_size - len(OPENING_DIRECTIVES))
                batch_directives.extend(extras)
                random.shuffle(batch_directives)

            # Sample capitalization styles for this batch (weighted random)
            cap_texts = [s[0] for s in CAPITALIZATION_STYLES]
            cap_weights = [s[1] for s in CAPITALIZATION_STYLES]
            batch_cap_styles = random.choices(cap_texts, weights=cap_weights, k=batch_size)

            tasks = [
                generate_single_review(i, neb_buffer, batch_directives[i - batch_start], batch_cap_styles[i - batch_start])
                for i in range(batch_start, batch_end)
            ]

            # When PocketBase is active, wrap each task with per-review progress updates.
            # This gives smooth real-time progress as each LLM call completes (vs. per-batch jumps).
            if convex and convex.has_fast_updates and not progress_callback:
                _pb_completed = 0
                async def _with_progress(coro):
                    nonlocal _pb_completed
                    result = await coro
                    _pb_completed += 1
                    pf = 5 + (batch_start + _pb_completed) / total_reviews * 94
                    await convex.update_progress(job_id, min(int(pf), 99), "AML")
                    return result
                tasks = [_with_progress(t) for t in tasks]

            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in batch_results:
                if isinstance(result, Exception):
                    error_count += 1
                    print(f"[Generation] Review exception: {type(result).__name__}: {result}", flush=True)
                    continue

                if result["error"]:
                    error_count += 1
                    last_error_msg = result["error_msg"]
                    continue

                if result["malformed"]:
                    malformed_count += 1
                    continue

                parsed_review = result["parsed_review"]
                if parsed_review:
                    sentences = parsed_review["sentences"]
                    for sent in sentences:
                        for opinion in sent.get("opinions", []):
                            target = opinion.get("target")
                            if target and target != "NULL":
                                target = target.strip()
                                words = target.split()
                                if len(words) > 1:
                                    target = words[-1]
                                target = target.lower()
                                text = sent.get("text", "")
                                idx = text.lower().find(target)
                                if idx >= 0:
                                    opinion["target"] = target
                                    opinion["from"] = idx
                                    opinion["to"] = idx + len(target)
                                else:
                                    opinion["target"] = "NULL"
                                    opinion["from"] = 0
                                    opinion["to"] = 0
                            else:
                                opinion["target"] = "NULL"
                                opinion["from"] = 0
                                opinion["to"] = 0

                    full_text = " ".join(s["text"] for s in sentences)
                    reviews.append({
                        "id": f"aml-{result['aml_number']}",
                        "review_text": full_text,
                        "sentences": sentences,
                        "assigned": {
                            "polarity_distribution": {
                                "positive": int(polarity_dist["positive"] * 100),
                                "neutral": int(polarity_dist["neutral"] * 100),
                                "negative": int(polarity_dist["negative"] * 100),
                            },
                            "age": result["reviewer_age"],
                            "sex": result["reviewer_sex"],
                            "num_sentences": result["num_sentences"],
                            "temperature": round(result["temperature"], 2),
                        },
                    })
                    generated_count += 1
                    total_sentences_generated += len(sentences)

                    if count_mode == 'sentences' and target_sentences and total_sentences_generated >= target_sentences:
                        print(f"[Generation] Target reached: {total_sentences_generated} sentences")
                        if convex:
                            await convex.add_log(job_id, "INFO", "AML", f"Target reached: {total_sentences_generated} sentences")
                        stop_generation = True
                        break

                elif result["review_text"]:
                    reviews.append({
                        "id": f"aml-{result['aml_number']}",
                        "review_text": result["review_text"].strip(),
                        "sentences": [],
                        "assigned": {
                            "polarity_distribution": {
                                "positive": int(polarity_dist["positive"] * 100),
                                "neutral": int(polarity_dist["neutral"] * 100),
                                "negative": int(polarity_dist["negative"] * 100),
                            },
                            "age": result["reviewer_age"],
                            "sex": result["reviewer_sex"],
                            "num_sentences": result["num_sentences"],
                            "temperature": round(result["temperature"], 2),
                        },
                    })
                    generated_count += 1
                    total_sentences_generated += result["num_sentences"]

            # Update NEB buffer and vocab tracker with this batch's successful reviews
            batch_review_texts = []
            successful_count = 0
            error_in_batch = 0
            for result in batch_results:
                if isinstance(result, Exception):
                    error_in_batch += 1
                    continue
                if not result.get("error") and not result.get("malformed"):
                    parsed_review = result.get("parsed_review")
                    if parsed_review and "sentences" in parsed_review:
                        successful_count += 1
                        texts = [s.get("text", "") for s in parsed_review["sentences"] if s.get("text")]
                        if texts:
                            review_text = " ".join(texts)
                            batch_review_texts.append(review_text)
                            if rgm_enabled:
                                vocab_tracker.update(review_text)

            if neb_buffer is not None:
                print(f"[NEB DEBUG] Batch {batch_start}-{batch_end}: {successful_count} successful, {error_in_batch} errors, {len(batch_review_texts)} texts collected", flush=True)
                if batch_review_texts:
                    neb_buffer.add_batch(batch_review_texts)
                    print(f"[NEB] Buffer updated: {len(neb_buffer)} reviews in memory", flush=True)
                else:
                    print(f"[NEB DEBUG] No review texts to add to buffer!", flush=True)

            if vocab_tracker.review_count > 0 and vocab_tracker.review_count % 50 == 0:
                stats = vocab_tracker.get_stats()
                print(f"[VDT] Tracked {stats['reviews_tracked']} reviews, {stats['unique_bigrams']} unique bigrams, {stats['overused_count']} overused phrases", flush=True)

            # Update progress after batch
            elapsed = time.time() - start_time
            if count_mode == 'sentences' and target_sentences:
                progress_float = 5 + min(total_sentences_generated / target_sentences, 1.0) * 94
            else:
                progress_float = 5 + batch_end / total_reviews * 94
            progress = min(int(progress_float), 99)
            rate = generated_count / elapsed * 60 if elapsed > 0 else 0

            if progress_callback:
                # Multi-model mode: use per-model callback instead of global progress
                await progress_callback(generated_count, error_count, progress)
            elif convex:
                await convex.update_progress(job_id, progress, "AML")
                await convex.update_generated_count(job_id, generated_count)
                # Update sentence count for sentence mode UI tracking
                if count_mode == 'sentences':
                    await convex.update_generated_sentences(job_id, total_sentences_generated)
                if error_count > 0:
                    await convex.update_failed_count(job_id, error_count)

            elapsed_str = f"{int(elapsed)}s" if elapsed < 60 else f"{int(elapsed//60)}m {int(elapsed%60)}s"
            print(f"[Generation] Progress: {generated_count}/{total_reviews} ({progress}%) | {rate:.1f}/min")
            if convex:
                await convex.add_log(job_id, "INFO", "AML", f"Progress: {generated_count}/{total_reviews} ({progress}%) | {elapsed_str} | {rate:.1f}/min")

            batch_start = batch_end

        # Sort reviews by ID to ensure consistent ordering
        reviews.sort(key=lambda r: r["id"])

        # ========================================
        # Save dataset (always save JSONL + user-selected formats)
        # ========================================
        if convex:
            await convex.update_progress(job_id, 99, "Saving dataset...")
            await convex.add_log(job_id, "INFO", "Dataset", "Saving dataset files...")

        # Get selected output formats (default to jsonl if none specified)
        output_formats = config.generation.output_formats or ["jsonl"]
        # Always ensure JSONL is included (needed for conformity calculation and internal processing)
        if "jsonl" not in output_formats:
            output_formats = list(output_formats) + ["jsonl"]
        saved_formats = []

        # Determine dataset mode for export
        dataset_mode = getattr(config.generation, "dataset_mode", "explicit")

        def build_review_sentences(review, mode):
            """Build sentence list with proper target/offset handling for the given mode."""
            sentences = review.get("sentences", [])
            if not sentences:
                # Fallback: single sentence with review text, generic opinion
                return [{
                    "text": review["review_text"],
                    "opinions": [{"target": "NULL", "category": "PRODUCT#QUALITY", "polarity": review["assigned"]["polarity"], "from": 0, "to": 0}]
                }]
            if mode == "implicit":
                # Strip targets and offsets for implicit mode
                result = []
                for sent in sentences:
                    new_opinions = []
                    for op in sent.get("opinions", []):
                        new_opinions.append({"target": "NULL", "category": op["category"], "polarity": op["polarity"], "from": 0, "to": 0})
                    result.append({"text": sent["text"], "opinions": new_opinions})
                return result
            else:
                # Explicit mode: ensure all opinions have from/to
                result = []
                for sent in sentences:
                    new_opinions = []
                    for op in sent.get("opinions", []):
                        new_opinions.append({
                            "target": op.get("target", "NULL"),
                            "category": op["category"],
                            "polarity": op["polarity"],
                            "from": op.get("from", 0),
                            "to": op.get("to", 0),
                        })
                    result.append({"text": sent["text"], "opinions": new_opinions})
                return result

        # Resolve file naming prefix from target_prefix config (falls back to job directory name)
        _target_prefix = getattr(config.generation, 'target_prefix', None) or ""
        if not _target_prefix.strip():
            # Derive from job directory name: "{job_id}-{sanitized-name}" → use the sanitized name part
            dir_name = job_path.name  # e.g. "abc123-my-hotel-reviews"
            parts = dir_name.split("-", 1)
            _target_prefix = parts[1] if len(parts) > 1 else dir_name
        _file_base = _target_prefix.strip()

        def save_dataset_files(reviews_data, mode, suffix=""):
            """Save all selected format files for a given mode."""
            fmt_saved = []
            file_suffix = f"-{suffix}" if suffix else ""
            # Multi-model: prefix with --{model_tag} (e.g. reviews--gemini-3-flash.jsonl)
            model_prefix = f"--{model_tag}" if model_tag else ""

            # Save JSONL only if selected by user
            if "jsonl" in output_formats:
                jsonl_path = dataset_path / f"{_file_base}{model_prefix}{file_suffix}.jsonl"
                with open(jsonl_path, "w", encoding="utf-8") as f:
                    for idx, review in enumerate(reviews_data):
                        sentences = build_review_sentences(review, mode)
                        # Handle polarity - can be distribution object or single value
                        assigned = review.get("assigned", {})
                        polarity_data = assigned.get("polarity_distribution") or assigned.get("polarity", {})
                        jsonl_entry = {
                            "id": str(idx),
                            "sentences": [],
                            "metadata": {
                                "assigned_polarity": polarity_data,
                                "age": assigned.get("age"),
                                "sex": assigned.get("sex"),
                            }
                        }
                        for s_idx, sent in enumerate(sentences):
                            jsonl_entry["sentences"].append({
                                "id": f"{idx}:{s_idx}",
                                "text": sent["text"],
                                "opinions": sent["opinions"],
                            })
                        f.write(json.dumps(jsonl_entry, ensure_ascii=False) + "\n")
                fmt_saved.append("JSONL")
                print(f"[Generation] Saved JSONL ({mode}): {jsonl_path}")

            # Save CSV (if selected) - flattened per-opinion
            if "csv" in output_formats:
                import csv as csv_module
                csv_path = dataset_path / f"{_file_base}{model_prefix}{file_suffix}.csv"
                with open(csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv_module.writer(f)
                    writer.writerow(["review_id", "sentence_id", "text", "target", "category", "polarity", "from", "to"])
                    for idx, review in enumerate(reviews_data):
                        sentences = build_review_sentences(review, mode)
                        for s_idx, sent in enumerate(sentences):
                            for op in sent["opinions"]:
                                writer.writerow([
                                    idx,
                                    f"{idx}:{s_idx}",
                                    sent["text"],
                                    op["target"],
                                    op["category"],
                                    op["polarity"],
                                    op["from"],
                                    op["to"],
                                ])
                fmt_saved.append("CSV")
                print(f"[Generation] Saved CSV ({mode}): {csv_path}")

            # Save SemEval XML (if selected)
            if "semeval_xml" in output_formats or "semeval" in output_formats:
                xml_path = dataset_path / f"{_file_base}{model_prefix}{file_suffix}.xml"
                xml_content = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
                xml_content += '<Reviews>\n'
                for idx, review in enumerate(reviews_data):
                    sentences = build_review_sentences(review, mode)
                    xml_content += f'  <Review rid="{idx}">\n'
                    xml_content += '    <sentences>\n'
                    for s_idx, sent in enumerate(sentences):
                        text_escaped = sent["text"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
                        xml_content += f'      <sentence id="{idx}:{s_idx}">\n'
                        xml_content += f'        <text>{text_escaped}</text>\n'
                        xml_content += '        <Opinions>\n'
                        for op in sent["opinions"]:
                            target_escaped = op["target"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
                            xml_content += f'          <Opinion target="{target_escaped}" category="{op["category"]}" polarity="{op["polarity"]}" from="{op["from"]}" to="{op["to"]}" />\n'
                        xml_content += '        </Opinions>\n'
                        xml_content += '      </sentence>\n'
                    xml_content += '    </sentences>\n'
                    xml_content += '  </Review>\n'
                xml_content += '</Reviews>\n'
                with open(xml_path, "w", encoding="utf-8") as f:
                    f.write(xml_content)
                fmt_saved.append("SemEval XML")
                print(f"[Generation] Saved SemEval XML ({mode}): {xml_path}")

            return fmt_saved

        # Save based on dataset mode
        if dataset_mode == "both":
            # Generate explicit first, then strip to implicit
            saved_formats.extend(save_dataset_files(reviews, "explicit", suffix="explicit"))
            saved_formats.extend(save_dataset_files(reviews, "implicit", suffix="implicit"))
        else:
            saved_formats.extend(save_dataset_files(reviews, dataset_mode))

        if convex and saved_formats:
            await convex.add_log(job_id, "INFO", "Dataset", f"Saved formats: {', '.join(saved_formats)}")

        # ========================================
        # Check for failure conditions
        # ========================================
        total_elapsed = time.time() - start_time
        final_error_rate = error_count / total_reviews if total_reviews > 0 else 1.0
        final_rate = generated_count / total_elapsed * 60 if total_elapsed > 0 else 0

        # Format elapsed time
        if total_elapsed < 60:
            elapsed_str = f"{total_elapsed:.1f}s"
        elif total_elapsed < 3600:
            elapsed_str = f"{int(total_elapsed//60)}m {int(total_elapsed%60)}s"
        else:
            elapsed_str = f"{int(total_elapsed//3600)}h {int((total_elapsed%3600)//60)}m"

        # Fail if no reviews generated or error rate > 50%
        if generated_count == 0:
            fail_msg = f"Generation failed: No reviews were generated. All {total_reviews} attempts failed."
            print(f"[Generation] {fail_msg}")
            if convex:
                await convex.update_progress(job_id, 100, "Failed")  # Set to 100% so progress bar shows completion
                await convex.add_log(job_id, "ERROR", "AML", fail_msg)
                if last_error_msg:
                    await convex.add_log(job_id, "ERROR", "AML", f"Last error: {last_error_msg[:200]}")
                await convex.fail_job(job_id, fail_msg)
            return {
                "status": "failed",
                "jobId": job_id,
                "error": fail_msg,
            }

        if final_error_rate > 0.5:
            fail_msg = f"Generation failed: Too many errors ({error_count}/{total_reviews} failed, {final_error_rate:.0%} error rate)."
            print(f"[Generation] {fail_msg}")
            if convex:
                await convex.update_progress(job_id, 100, "Failed")  # Set to 100% so progress bar shows completion
                await convex.add_log(job_id, "ERROR", "AML", fail_msg)
                if last_error_msg:
                    await convex.add_log(job_id, "ERROR", "AML", f"Last error: {last_error_msg[:200]}")
                await convex.fail_job(job_id, fail_msg)
            return {
                "status": "failed",
                "jobId": job_id,
                "error": fail_msg,
            }

        # ========================================
        # Complete with detailed summary
        # ========================================
        if convex:
            await convex.update_progress(job_id, 100, "Complete")
            await convex.update_generated_count(job_id, generated_count)
            # Final sentence count update for sentence mode
            if count_mode == 'sentences':
                await convex.update_generated_sentences(job_id, total_sentences_generated)

            # Log summary statistics
            await convex.add_log(job_id, "INFO", "Summary", "=" * 40)
            await convex.add_log(job_id, "INFO", "Summary", f"Generation completed successfully!")
            await convex.add_log(job_id, "INFO", "Summary", f"Reviews: {generated_count}/{total_reviews} ({generated_count/total_reviews*100:.1f}%)")
            await convex.add_log(job_id, "INFO", "Summary", f"Duration: {elapsed_str} ({final_rate:.1f} reviews/min)")

            if error_count > 0:
                await convex.add_log(job_id, "WARN", "Summary", f"Errors: {error_count} failed ({final_error_rate:.1%} error rate)")

            if malformed_count > 0:
                await convex.add_log(job_id, "WARN", "Summary", f"Malformed JSON: {malformed_count} reviews fell back to plain text")

            # Polarity breakdown of actual generated reviews (from sentences)
            pol_counts = {"positive": 0, "neutral": 0, "negative": 0}
            for r in reviews:
                # Count sentence-level polarities from structured data
                for sent in r.get("sentences", []):
                    for op in sent.get("opinions", []):
                        pol = op.get("polarity", "").lower()
                        if pol in pol_counts:
                            pol_counts[pol] += 1

            await convex.add_log(
                job_id, "INFO", "Summary",
                f"Polarity: {pol_counts['positive']} pos, {pol_counts['neutral']} neu, {pol_counts['negative']} neg"
            )
            await convex.add_log(job_id, "INFO", "Summary", f"Output: {dataset_path}")
            await convex.add_log(job_id, "INFO", "Summary", "=" * 40)

            # Only complete job if this is a standalone generation (not part of pipeline)
            if should_complete_job:
                await convex.complete_job(job_id)

        print(f"[Generation] Complete! {generated_count} reviews saved to {dataset_path} in {elapsed_str}")

        return {
            "status": "completed",
            "jobId": job_id,
            "generatedCount": generated_count,
            "malformedCount": malformed_count,
            "errorCount": error_count,
            "datasetPath": str(dataset_path),
            "reviews": reviews,
        }

    except Exception as e:
        error_msg = str(e)
        print(f"[Generation] Error: {error_msg}")
        if convex:
            await convex.add_log(job_id, "ERROR", "Generation", f"Generation failed: {error_msg}")
            await convex.fail_job(job_id, error_msg)
        raise


@app.post("/api/generate-job")
async def generate_job(request: GenerationRequest, background_tasks: BackgroundTasks):
    """
    Start the generation phase of a job in the background.

    This expects the job to already have composition files created.
    """
    print(f"[API] Starting generation for job: {request.jobId}")
    print(f"[API] Job directory: {request.jobDir}")
    print(f"[API] Convex URL: {request.convexUrl or 'NOT PROVIDED'}")
    print(f"[API] Convex Token: {'PROVIDED (' + str(len(request.convexToken or '')) + ' chars)' if request.convexToken else 'NOT PROVIDED'}")

    background_tasks.add_task(
        execute_generation,
        request.jobId,
        request.jobDir,
        request.config,
        request.apiKey,
        request.convexUrl,
        request.convexToken,
    )
    return {"status": "started", "jobId": request.jobId}


class DeleteJobDirRequest(BaseModel):
    """Request to delete a job directory."""
    jobDir: str  # Full path to the job directory


@app.post("/api/delete-job-dir")
async def delete_job_directory(request: DeleteJobDirRequest):
    """
    Delete a job directory and all its contents.

    This is called when a job is deleted from the frontend.
    """
    import shutil
    from pathlib import Path

    job_dir = Path(request.jobDir)

    # Safety checks
    if not job_dir.exists():
        print(f"[API] Job directory does not exist: {job_dir}")
        return {"status": "not_found", "message": "Directory does not exist"}

    # Ensure the directory is within the expected jobs directory structure
    # Allow both ./jobs and /app/jobs (container path)
    job_dir_resolved = job_dir.resolve()
    valid_prefixes = [
        Path("./jobs").resolve(),
        Path("/app/jobs").resolve(),
    ]

    is_valid_path = any(
        str(job_dir_resolved).startswith(str(prefix))
        for prefix in valid_prefixes
    )

    if not is_valid_path:
        print(f"[API] Invalid job directory path (not in jobs dir): {job_dir}")
        raise HTTPException(
            status_code=400,
            detail="Invalid job directory path - must be within jobs directory"
        )

    try:
        shutil.rmtree(job_dir)
        print(f"[API] Deleted job directory: {job_dir}")
        return {"status": "deleted", "path": str(job_dir)}
    except Exception as e:
        print(f"[API] Failed to delete job directory: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete directory: {str(e)}"
        )


# ============================================================
# New endpoints for redesigned job creation/pipeline flow
# ============================================================


class CheckContextsRequest(BaseModel):
    """Request to check if context files exist in a job directory."""
    jobDir: str


@app.post("/api/check-contexts")
async def check_contexts(request: CheckContextsRequest):
    """Check if a job directory has context files available."""
    from pathlib import Path

    contexts_dir = Path(request.jobDir) / "contexts"

    has_subject = (contexts_dir / "subject-context.json").exists()
    has_reviewers = (contexts_dir / "reviewers-context.json").exists()
    has_attributes = (contexts_dir / "attributes-context.json").exists()

    return {
        "hasContexts": has_subject and has_reviewers and has_attributes,
        "subject": has_subject,
        "reviewers": has_reviewers,
        "attributes": has_attributes,
    }


class LoadJobConfigRequest(BaseModel):
    """Request to load config.json from a job directory."""
    jobDir: str


@app.post("/api/load-job-config")
async def load_job_config(request: LoadJobConfigRequest):
    """
    Load config.json from a job directory.

    Used when reusing composition from an existing job to:
    1. Preview the composition settings
    2. Pre-fill the form with compatible settings
    """
    from pathlib import Path
    import json

    config_path = Path(request.jobDir) / "config.json"
    if not config_path.exists():
        raise HTTPException(status_code=404, detail="config.json not found in job directory")

    try:
        with open(config_path, encoding="utf-8") as f:
            config_data = json.load(f)
        return {"config": config_data, "path": str(config_path)}
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON in config.json: {str(e)}")


class CreateContextsOnlyRequest(BaseModel):
    """Request to create only RGM + ACM contexts (no LLM, instant)."""
    jobId: str
    jobName: str
    config: JobConfig
    jobsDirectory: str = "./jobs"


@app.post("/api/create-contexts-only")
async def create_contexts_only(request: CreateContextsOnlyRequest):
    """
    Create only reviewer and attributes contexts (RGM + ACM).

    This is instant (no LLM calls needed) and runs during job creation.
    SIL/MAV subject context is deferred to pipeline execution.
    """
    job_paths = create_job_directory(
        request.jobsDirectory, request.jobId, request.jobName
    )

    # Check ablation settings for age and sex
    age_enabled = getattr(request.config.ablation, 'age_enabled', True) if request.config.ablation else True
    sex_enabled = getattr(request.config.ablation, 'sex_enabled', True) if request.config.ablation else True

    # RGM: Save reviewer context (just the distribution specs)
    # Respect ablation settings: None for age_range if disabled, 100% unspecified for sex if disabled
    reviewers_context = {
        "age_range": request.config.reviewer_profile.age_range if age_enabled else None,
        "sex_distribution": request.config.reviewer_profile.sex_distribution if sex_enabled else {"male": 0.0, "female": 0.0, "unspecified": 1.0},
        "additional_context": request.config.reviewer_profile.additional_context,
        "review_count": request.config.generation.count if request.config.generation else 1000,
    }

    reviewers_context_path = Path(job_paths["contexts"]) / "reviewers-context.json"
    with open(reviewers_context_path, "w") as f:
        json.dump(reviewers_context, f, indent=2)

    # ACM: Save attributes context (just the distribution specs)
    attributes_context = {
        "polarity": {
            "positive": request.config.attributes_profile.polarity.positive,
            "neutral": request.config.attributes_profile.polarity.neutral,
            "negative": request.config.attributes_profile.polarity.negative,
        },
        "noise": {
            "typo_rate": request.config.attributes_profile.noise.typo_rate,
            "colloquialism": request.config.attributes_profile.noise.colloquialism,
            "grammar_errors": request.config.attributes_profile.noise.grammar_errors,
            "preset": request.config.attributes_profile.noise.preset,
        },
        "length_range": request.config.attributes_profile.length_range,
        "edge_lengths": getattr(request.config.attributes_profile, 'edge_lengths', None) and getattr(request.config.attributes_profile, 'edge_lengths').model_dump(),
    }

    attributes_context_path = Path(job_paths["contexts"]) / "attributes-context.json"
    with open(attributes_context_path, "w") as f:
        json.dump(attributes_context, f, indent=2)

    print(f"[API] Created contexts-only for job {request.jobId}: {job_paths['root']}")

    return {
        "status": "created",
        "jobId": request.jobId,
        "jobDir": job_paths["root"],
        "reviewerContext": reviewers_context,
        "attributesContext": attributes_context,
    }


class HeuristicTargetDataset(BaseModel):
    """A single target dataset size for heuristic generation."""
    targetMode: str = "sentences"  # "reviews" or "sentences"
    targetValue: int = 100
    reviewsPerBatch: int = 25
    requestSize: int = 3
    totalRuns: int = 1
    runsMode: str = "parallel"  # "parallel" or "sequential"


class HeuristicConfig(BaseModel):
    """Configuration for heuristic (baseline) prompting method."""
    prompt: str
    useFormatPrompt: Optional[bool] = True  # Whether to append format instructions
    formatPrompt: Optional[str] = None  # Custom format prompt (ABSA JSON instructions)
    targetMode: str = "sentences"  # "reviews" or "sentences" (legacy, use targets[0])
    targetValue: int = 100
    reviewsPerBatch: int = 25
    requestSize: int = 3  # Number of parallel batch requests
    avgSentencesPerReview: str = "5"  # e.g., "5" or "4-7" (passed verbatim to prompt)
    model: str = ""
    outputFormat: str = "semeval_xml"  # "semeval_xml", "jsonl", "csv"
    totalRuns: int = 1  # Number of times to run generation (for research variability assessment)
    parallelRuns: bool = True  # Run all runs concurrently
    knowledgeSourceJobId: Optional[str] = None  # Job ID to import SIL knowledge from
    # Multi-target dataset support
    targetPrefix: Optional[str] = None  # File naming prefix (e.g., "rq1-heuristic")
    targets: Optional[list[HeuristicTargetDataset]] = None  # Multiple target dataset sizes
    parallelTargets: bool = True  # Run target datasets concurrently
    # Multi-model support
    models: Optional[list[str]] = None  # Multiple generation models (overrides `model`)
    parallelModels: bool = True  # Run models concurrently

    def get_effective_targets(self) -> list[HeuristicTargetDataset]:
        """Return targets array, synthesizing from legacy fields if needed."""
        if self.targets:
            return self.targets
        return [HeuristicTargetDataset(
            targetMode=self.targetMode,
            targetValue=self.targetValue,
            reviewsPerBatch=self.reviewsPerBatch,
            requestSize=self.requestSize,
            totalRuns=self.totalRuns,
            runsMode="parallel" if self.parallelRuns else "sequential",
        )]

    def get_effective_models(self) -> list[str]:
        """Return models list, falling back to single model field."""
        if self.models:
            return [m for m in self.models if m]
        return [self.model] if self.model else []


def _DEFAULT_HEURISTIC_FORMAT_PROMPT() -> str:
    """Default ABSA JSON format instructions appended to heuristic prompts."""
    return """

Return your response as a JSON array with ABSA (Aspect-Based Sentiment Analysis) format.
Each review should have sentences, and each sentence should have aspect-level opinions.

Structure:
[
  {
    "sentences": [
      {
        "text": "The food was delicious and fresh.",
        "opinions": [
          {"target": "food", "category": "FOOD#QUALITY", "polarity": "positive", "from": 4, "to": 8}
        ]
      },
      {
        "text": "However, the service was quite slow.",
        "opinions": [
          {"target": "service", "category": "SERVICE#GENERAL", "polarity": "negative", "from": 13, "to": 20}
        ]
      }
    ]
  },
  ...
]

Rules:
- "text": The sentence text (non-empty string)
- "target": MUST be a SINGLE WORD (noun) in lowercase that appears exactly in the text
- "category": Use categories mentioned in the prompt above
- "polarity": One of "positive", "negative", "neutral"
- "from": Character offset where target starts in the text (0-indexed)
- "to": Character offset where target ends (exclusive, NOT including any character after the word)

Important:
- The target MUST be a single lowercase word, NOT a phrase (e.g., "food" not "the food", "service" not "great service")
- The "from" and "to" must be accurate character positions for the target substring
- A sentence can have multiple opinions about different aspects
- A sentence can have zero opinions if it's neutral/general
Return ONLY the JSON array, no other text, no markdown code blocks."""


def _format_knowledge_for_heuristic(subject_context: dict) -> str:
    """Format SIL subject-context.json as a knowledge section for heuristic prompts."""
    sections = []

    sections.append("--- DOMAIN KNOWLEDGE (verified facts from CERA composition) ---")

    if subject_context.get("characteristics"):
        sections.append("\nVERIFIED DOMAIN FACTS (use ONLY these when mentioning specific technical details, specs, prices, or features — do NOT invent specifications or brand claims outside this list):")
        for fact in subject_context["characteristics"]:
            sections.append(f"- {fact}")

    if subject_context.get("positives"):
        sections.append("\nPOSITIVE FEATURES TO WEAVE IN (for positive/mixed reviews):")
        for pos in subject_context["positives"]:
            sections.append(f"- {pos}")

    if subject_context.get("negatives"):
        sections.append("\nNEGATIVE FEATURES TO WEAVE IN (for negative/mixed reviews):")
        for neg in subject_context["negatives"]:
            sections.append(f"- {neg}")

    return "\n".join(sections)


class PipelineRequest(BaseModel):
    """Request to run the full pipeline (selected phases sequentially)."""
    jobId: str
    jobName: str
    config: Optional[JobConfig] = None  # Optional for heuristic method (which doesn't use CERA config)
    phases: list[str]  # ["composition", "generation", "evaluation"]
    apiKey: str
    tavilyApiKey: Optional[str] = None
    jobsDirectory: str = "./jobs"
    convexUrl: Optional[str] = None
    convexToken: Optional[str] = None
    evaluationConfig: Optional[dict] = None
    datasetFile: Optional[str] = None  # For EVAL-only jobs
    reusedFromJobDir: Optional[str] = None  # Source job directory for composition reuse
    referenceDataset: Optional[dict] = None  # Reference dataset info for config.json
    # Heuristic method fields (RQ1 baseline)
    method: str = "cera"  # "cera" or "heuristic"
    heuristicConfig: Optional[HeuristicConfig] = None
    # Token usage: pre-job RDE records from context extraction
    rdeUsage: Optional[list[dict]] = None


def _recover_truncated_json_array(content: str) -> list | None:
    """Try to recover complete objects from a truncated JSON array.

    When the LLM output is cut off mid-JSON, this finds the last complete
    top-level object in the array and returns all complete objects.
    """
    import json
    if not content or not content.strip().startswith('['):
        return None

    # Find the position of the last complete object by looking for "},\n  {"
    # or "}\n]" patterns working backwards
    last_complete = -1
    depth = 0
    in_string = False
    escape = False

    for i, ch in enumerate(content):
        if escape:
            escape = False
            continue
        if ch == '\\' and in_string:
            escape = True
            continue
        if ch == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 1:  # Closed a top-level object inside the array
                last_complete = i

    if last_complete <= 0:
        return None

    # Slice up to the last complete object and close the array
    truncated = content[:last_complete + 1].rstrip().rstrip(',') + '\n]'
    try:
        result = json.loads(truncated)
        if isinstance(result, list) and len(result) > 0:
            return result
    except json.JSONDecodeError:
        pass
    return None


async def execute_heuristic_pipeline(
    job_id: str,
    job_name: str,
    heuristic_config: HeuristicConfig,
    api_key: str,
    jobs_directory: str,
    convex_url: Optional[str],
    convex_token: Optional[str],
    evaluation_config: Optional[dict],
    rde_usage: Optional[list[dict]] = None,
):
    """
    Execute heuristic (baseline) pipeline for RQ1 comparison.

    Unlike CERA, this skips composition and uses direct LLM prompting.
    - No SIL (web search grounding)
    - No MAV (multi-agent verification)
    - No RGM (demographic sampling)
    - No noise injection
    - Just batched LLM calls with prompt template
    """
    import json
    import math
    import re
    from pathlib import Path

    convex = None
    if convex_url and convex_token:
        convex = ConvexClient(convex_url, convex_token, pocketbase_url=os.environ.get("POCKETBASE_URL"))

    try:
        # Create job directory
        job_paths = create_job_directory(jobs_directory, job_id, job_name)

        # Initialize token usage tracker
        from cera.llm.usage import UsageTracker
        usage_tracker = UsageTracker()
        tokens_json_path = Path(job_paths["reports"]) / "tokens.json"

        # Import pre-job RDE usage records if available
        if rde_usage:
            usage_tracker.import_records(rde_usage)
            print(f"[Tokens] Imported {len(rde_usage)} RDE usage records")

        def _save_tokens_incremental():
            """Incrementally save tokens.json (safe to call anytime)."""
            try:
                if usage_tracker.total_tokens > 0:
                    usage_tracker.save_tokens_json(tokens_json_path, job_id, "heuristic")
            except Exception as _e:
                print(f"[Tokens] Warning: Failed to save tokens.json: {_e}")

        # Multi-run support
        total_runs = heuristic_config.totalRuns if heuristic_config.totalRuns else 1

        if convex:
            if total_runs > 1:
                await convex.add_log(job_id, "INFO", "Heuristic", f"Starting heuristic generation pipeline ({total_runs} runs)...")
            else:
                await convex.add_log(job_id, "INFO", "Heuristic", "Starting heuristic generation pipeline...")

            # Early GPU/CPU detection - set immediately so UI shows the badge
            try:
                import torch
                if torch.cuda.is_available():
                    device_name = torch.cuda.get_device_name(0)
                    await convex.set_evaluation_device(job_id, "GPU", device_name)
                    await convex.add_log(job_id, "INFO", "Heuristic", f"GPU detected: {device_name}")
                else:
                    await convex.set_evaluation_device(job_id, "CPU")
            except ImportError:
                await convex.set_evaluation_device(job_id, "CPU")

        # Parse avg sentences for numeric calculations (e.g. "4-7" → 5.5, "5" → 5)
        avg_sentences_str = heuristic_config.avgSentencesPerReview
        range_match = re.match(r'^(\d+)\s*-\s*(\d+)$', avg_sentences_str)
        if range_match:
            avg_sentences_num = (int(range_match.group(1)) + int(range_match.group(2))) / 2
        else:
            try:
                avg_sentences_num = float(avg_sentences_str)
            except ValueError:
                avg_sentences_num = 5.0

        # Inject knowledge from source CERA job if provided
        knowledge_suffix = ""
        if heuristic_config.knowledgeSourceJobId:
            source_job_id = heuristic_config.knowledgeSourceJobId
            source_dirs = list(Path(jobs_directory).glob(f"{source_job_id}-*"))
            if source_dirs:
                context_file = source_dirs[0] / "contexts" / "subject-context.json"
                if context_file.exists():
                    with open(context_file) as f:
                        source_context = json.load(f)
                    knowledge_suffix = "\n\n" + _format_knowledge_for_heuristic(source_context)
                    if convex:
                        fact_count = len(source_context.get("characteristics", []))
                        await convex.add_log(job_id, "INFO", "Heuristic", f"Injected {fact_count} domain facts from knowledge source job")
                else:
                    if convex:
                        await convex.add_log(job_id, "WARN", "Heuristic", f"Knowledge source job directory found but no subject-context.json")
            else:
                if convex:
                    await convex.add_log(job_id, "WARN", "Heuristic", f"Knowledge source job directory not found for {source_job_id}")

        # ========================================
        # Generation + Evaluation Pipeline
        # (Unified path for all target/model combinations)
        # ========================================
        effective_targets = heuristic_config.get_effective_targets()
        effective_models = heuristic_config.get_effective_models()
        is_multi_target = len(effective_targets) > 1
        is_multi_model = len(effective_models) > 1
        use_format_prompt = heuristic_config.useFormatPrompt if heuristic_config.useFormatPrompt is not None else True

        # Pre-detect local models and fetch settings once
        _has_any_local = any(m.startswith("local/") for m in effective_models)
        _local_endpoint, _local_api_key = "", ""
        if _has_any_local and convex:
            _local_endpoint, _local_api_key = await _get_local_llm_settings(convex)

        eval_lock = asyncio.Lock()  # GPU eval lock (shared)

        # Shared progress tracking for multi-target pipeline
        _mt_progress = {
            "completed_batches": 0,
            "total_batches": 0,
            "reviews_collected": 0,
            "current_target": "",
        }
        _mt_progress_lock = asyncio.Lock()

        # Calculate total batches across all targets/runs for progress bar
        for _t in effective_targets:
            if _t.targetMode == "reviews":
                _t_reviews = _t.targetValue
            else:
                _t_reviews = math.ceil(_t.targetValue / avg_sentences_num)
            _t_batches = math.ceil(_t_reviews / _t.reviewsPerBatch)
            _mt_progress["total_batches"] += _t_batches * _t.totalRuns * len(effective_models)

        async def _update_mt_progress(batches_done: int = 0, reviews_done: int = 0, target_label: str = ""):
            """Update shared progress and report to Convex."""
            async with _mt_progress_lock:
                _mt_progress["completed_batches"] += batches_done
                _mt_progress["reviews_collected"] += reviews_done
                if target_label:
                    _mt_progress["current_target"] = target_label
                if convex:
                    await convex.run_mutation("jobs:updateHeuristicProgress", {
                        "jobId": job_id,
                        "currentBatch": _mt_progress["completed_batches"],
                        "totalBatches": _mt_progress["total_batches"],
                        "reviewsCollected": _mt_progress["reviews_collected"],
                    })

        def _prepare_prompt(reviews_per_batch: int) -> str:
            """Prepare full prompt with target-specific substitutions."""
            p = heuristic_config.prompt
            p = p.replace("{review_count}", str(reviews_per_batch))
            p = p.replace("{avg_sentences}", avg_sentences_str)
            p = p.replace("{sentence_count}", str(int(reviews_per_batch * avg_sentences_num)))
            if use_format_prompt:
                if heuristic_config.formatPrompt:
                    p += "\n\n" + heuristic_config.formatPrompt
                else:
                    p += _DEFAULT_HEURISTIC_FORMAT_PROMPT()
            if knowledge_suffix:
                p += knowledge_suffix
            return p

        async def _run_target_model(target: HeuristicTargetDataset, model_id: str, target_paths: dict):
            """Generate + save for one target × one model (all runs)."""
            model_slug = model_id.split("/")[-1]

            if target.targetMode == "reviews":
                t_reviews = target.targetValue
            else:
                t_reviews = math.ceil(target.targetValue / avg_sentences_num)
            t_batches = math.ceil(t_reviews / target.reviewsPerBatch)
            t_prompt = _prepare_prompt(target.reviewsPerBatch)

            async def _gen_run(run_num: int) -> list:
                """Generate all batches for one run."""
                reviews = []
                sem = asyncio.Semaphore(target.requestSize)

                # Per-run progress tracking (shows individual progress bars per run)
                if convex and t_runs > 1:
                    await convex.run_mutation("jobs:updateRunProgress", {
                        "jobId": job_id, "run": run_num, "status": "generating",
                        "currentBatch": 0, "totalBatches": t_batches, "reviewsCollected": 0,
                    })

                async def _gen_batch(batch_num: int) -> list:
                    async with sem:
                        if convex:
                            await convex.add_log(job_id, "INFO", "Heuristic",
                                f"[{target.targetValue}][{model_slug}] Run {run_num} batch {batch_num}/{t_batches}")
                        retries = 0
                        content = None
                        while retries < 3:
                            try:
                                from cera.llm.openrouter import OpenRouterClient
                                _is_local, _actual_mid = _parse_local_model(model_id)
                                _llm_kw = dict(
                                    usage_tracker=usage_tracker, component="heuristic",
                                    target=str(target.targetValue),
                                    run=f"run{run_num}" if t_runs > 1 else "",
                                )
                                if _is_local:
                                    llm = OpenRouterClient(api_key=_local_api_key, base_url=_local_endpoint, **_llm_kw)
                                else:
                                    llm = OpenRouterClient(api_key=api_key, **_llm_kw)
                                content = await llm.chat(
                                    model=_actual_mid,
                                    messages=[{"role": "user", "content": t_prompt}],
                                    temperature=0.8,
                                    max_tokens=16384,
                                )
                                batch_reviews = _extract_json_from_llm(content, expected_type="array")
                                if not batch_reviews:
                                    raise ValueError("No JSON array")
                                await _update_mt_progress(batches_done=1, reviews_done=len(batch_reviews))
                                return batch_reviews
                            except Exception as e:
                                retries += 1
                                content_preview = repr(content[:500]) if content else "(empty)"
                                print(f"[Heuristic] [{target.targetValue}][{model_slug}] Batch {batch_num} attempt {retries}/3 failed: {e}")
                                print(f"[Heuristic] LLM response preview ({len(content) if content else 0} chars): {content_preview}")
                                if retries >= 3:
                                    if convex:
                                        await convex.add_log(job_id, "ERROR", "Heuristic",
                                            f"[{target.targetValue}][{model_slug}] Batch {batch_num} failed after 3 attempts")
                                    return []
                        return []

                for wave_start in range(1, t_batches + 1, target.requestSize):
                    wave_end = min(wave_start + target.requestSize, t_batches + 1)
                    wave_results = await asyncio.gather(*[_gen_batch(b) for b in range(wave_start, wave_end)])
                    for batch_reviews in wave_results:
                        for review in batch_reviews:
                            rid = f"{job_id}-r{run_num}-{len(reviews):05d}"
                            if "sentences" in review:
                                reviews.append({"id": rid, "sentences": review["sentences"], "method": "heuristic"})
                            else:
                                text = review.get("text", "")
                                polarity = review.get("polarity", "neutral")
                                reviews.append({"id": rid, "sentences": [{"text": text, "opinions": [
                                    {"target": "NULL", "category": "GENERAL", "polarity": polarity, "from": 0, "to": 0}
                                ]}], "method": "heuristic"})

                    # Update per-run progress after each wave
                    if convex and t_runs > 1:
                        completed_batches = min(wave_end - 1, t_batches)
                        await convex.run_mutation("jobs:updateRunProgress", {
                            "jobId": job_id, "run": run_num, "status": "generating",
                            "currentBatch": completed_batches, "totalBatches": t_batches,
                            "reviewsCollected": len(reviews),
                        })

                # Normalize targets
                for review in reviews:
                    for sentence in review.get("sentences", []):
                        text = sentence.get("text", "")
                        for opinion in sentence.get("opinions", []):
                            t = opinion.get("target")
                            if t and t != "NULL":
                                t = t.strip().split()[-1].lower()
                                idx = text.lower().find(t)
                                if idx >= 0:
                                    opinion["target"] = t
                                    opinion["from"] = idx
                                    opinion["to"] = idx + len(t)
                                else:
                                    opinion["target"] = "NULL"
                                    opinion["from"] = 0
                                    opinion["to"] = 0
                # Mark run as completed
                if convex and t_runs > 1:
                    await convex.run_mutation("jobs:updateRunProgress", {
                        "jobId": job_id, "run": run_num, "status": "completed",
                        "currentBatch": t_batches, "totalBatches": t_batches,
                        "reviewsCollected": len(reviews),
                    })

                return reviews

            # Execute runs (parallel or sequential)
            t_runs = target.totalRuns
            t_parallel = target.runsMode == "parallel"
            if t_parallel and t_runs > 1:
                run_results = await asyncio.gather(*[_gen_run(r) for r in range(1, t_runs + 1)])
            else:
                run_results = [await _gen_run(r) for r in range(1, t_runs + 1)]

            # Save datasets per run into run{N}/{model_slug}/ directories
            for run_num, reviews in enumerate(run_results, 1):
                if not reviews:
                    continue

                # Get the correct output directory for this run + model
                run_model_dir = Path(target_paths["runs"][run_num]["models"][model_id])

                # Save explicit + implicit
                for mode_suffix, implicit in [("explicit", False), ("implicit", True)]:
                    import copy as _copy
                    save_reviews = _copy.deepcopy(reviews) if implicit else reviews
                    if implicit:
                        for r in save_reviews:
                            for s in r.get("sentences", []):
                                for o in s.get("opinions", []):
                                    o["target"] = "NULL"
                                    o["from"] = 0
                                    o["to"] = 0

                    # Filename: {prefix}-{mode}.{ext} (run/model encoded in directory path)
                    file_stem = f"{_h_file_base}-{mode_suffix}"

                    # JSONL (always)
                    jsonl_path = run_model_dir / f"{file_stem}.jsonl"
                    with open(jsonl_path, "w") as f:
                        for r in save_reviews:
                            f.write(json.dumps(r, ensure_ascii=False) + "\n")

                    # SemEval XML
                    if heuristic_config.outputFormat == "semeval_xml":
                        xml_path = run_model_dir / f"{file_stem}.xml"
                        xml_content = ['<?xml version="1.0" encoding="UTF-8"?>', '<Reviews>']
                        for r in save_reviews:
                            xml_content.append(f'  <Review rid="{r["id"]}">')
                            xml_content.append('    <sentences>')
                            for s_idx, s in enumerate(r.get("sentences", [])):
                                sid = f"{r['id']}:{s_idx}"
                                xml_content.append(f'      <sentence id="{sid}">')
                                xml_content.append(f'        <text>{escape_xml(s.get("text", ""))}</text>')
                                xml_content.append('        <Opinions>')
                                for o in s.get("opinions", []):
                                    xml_content.append(
                                        f'          <Opinion target="{escape_xml(o.get("target", "NULL"))}" '
                                        f'category="{o.get("category", "")}" '
                                        f'polarity="{o.get("polarity", "neutral")}" '
                                        f'from="{o.get("from", 0)}" to="{o.get("to", 0)}"/>')
                                xml_content.append('        </Opinions>')
                                xml_content.append('      </sentence>')
                            xml_content.append('    </sentences>')
                            xml_content.append('  </Review>')
                        xml_content.append('</Reviews>')
                        with open(xml_path, "w", encoding="utf-8") as f:
                            f.write("\n".join(xml_content))

                    # CSV
                    elif heuristic_config.outputFormat == "csv":
                        import csv as csv_module
                        csv_path = run_model_dir / f"{file_stem}.csv"
                        with open(csv_path, "w", newline="", encoding="utf-8") as f:
                            writer = csv_module.DictWriter(f, fieldnames=[
                                "review_id", "sentence_id", "text", "target", "category", "polarity", "from", "to"
                            ])
                            writer.writeheader()
                            for r in save_reviews:
                                for s_idx, s in enumerate(r.get("sentences", [])):
                                    for o in s.get("opinions", []):
                                        writer.writerow({
                                            "review_id": r["id"],
                                            "sentence_id": f"{r['id']}_{s_idx}",
                                            "text": s.get("text", ""),
                                            "target": o.get("target", "NULL"),
                                            "category": o.get("category", ""),
                                            "polarity": o.get("polarity", "neutral"),
                                            "from": o.get("from", 0),
                                            "to": o.get("to", 0),
                                        })

                if convex:
                    await convex.add_log(job_id, "INFO", "Heuristic",
                        f"[{target.targetValue}][{model_slug}] Run {run_num} saved ({len(reviews)} reviews)")

            return run_results

        # Resolve file naming prefix (falls back to job directory name)
        _h_prefix = heuristic_config.targetPrefix or ""
        if not _h_prefix.strip():
            _job_dir_name = Path(job_paths["root"]).name
            _parts = _job_dir_name.split("-", 1)
            _h_prefix = _parts[1] if len(_parts) > 1 else _job_dir_name
        _h_file_base = _h_prefix.strip()

        # Orchestrate: targets × models
        if convex:
            target_labels = [f"{t.targetValue} {t.targetMode}" for t in effective_targets]
            model_labels = [m.split("/")[-1] for m in effective_models]
            await convex.add_log(job_id, "INFO", "Heuristic",
                f"Pipeline: {len(effective_targets)} target(s) ({', '.join(target_labels)}) × "
                f"{len(effective_models)} model(s) ({', '.join(model_labels)})")

        # Collect metrics across all targets for final aggregation
        _all_target_metrics = []  # list of metrics dicts from all evaluations

        async def _run_heuristic_target(target_idx, target):
            nonlocal _h_file_base
            # Create target directory
            target_paths = create_target_directory(
                job_paths["datasets"], target.targetValue, target.targetMode,
                target.totalRuns, effective_models,
            )
            target_job_paths = dict(job_paths)
            target_job_paths["dataset"] = target_paths["target"]
            target_job_paths["metrics"] = target_paths["metrics"]

            if convex:
                await convex.add_log(job_id, "INFO", "Heuristic",
                    f"[Target {target_idx+1}/{len(effective_targets)}] {target.targetValue} {target.targetMode}")

            # Run models for this target (parallel or sequential)
            parallel_models = heuristic_config.parallelModels

            async def _model_task(model_id):
                await _run_target_model(target, model_id, target_paths)

            if parallel_models and is_multi_model:
                await asyncio.gather(*[_model_task(m) for m in effective_models])
            else:
                for m in effective_models:
                    await _model_task(m)

            # Evaluation for this target
            if evaluation_config and evaluation_config.get("metrics"):
                if convex:
                    # Signal evaluation phase (idempotent — may be called per target)
                    try:
                        await convex.run_mutation("jobs:startEvaluation", {"id": job_id})
                    except Exception:
                        pass  # Already in evaluating state from another target
                    await convex.add_log(job_id, "INFO", "MDQA",
                        f"[Target {target.targetValue}] Starting evaluation...")

                # Find dataset files in run{N}/{model_slug}/ subdirectories
                target_dir = Path(target_paths["target"])
                target_eval_metrics = []  # list of (metrics_dict, dataset_file_path)
                for ext in [".jsonl"]:
                    for dataset_file_path in sorted(target_dir.glob(f"run*/**/*-explicit{ext}")):
                        async with eval_lock:
                            metrics_result = await execute_evaluation(
                                job_id=job_id,
                                job_paths=target_job_paths,
                                evaluation_config=evaluation_config,
                                dataset_file=str(dataset_file_path),
                                convex=convex,
                                save_files=False,
                            )
                            if metrics_result:
                                target_eval_metrics.append((metrics_result, str(dataset_file_path)))

                # Build per-run metrics list and extract model info from paths
                import re as _re
                metric_keys = ["bleu", "rouge_l", "bertscore", "moverscore", "distinct_1", "distinct_2", "self_bleu"]
                per_run_metrics = []
                # Also collect per-model data: model_slug -> {metrics_list, per_run}
                _h_per_model = {}
                for metrics_dict, file_path in target_eval_metrics:
                    # Extract run number from path like .../run1/model-slug/file.jsonl
                    run_match = _re.search(r'/run(\d+)/', file_path)
                    run_num = int(run_match.group(1)) if run_match else 1
                    # Extract model slug from path like .../run1/model-slug/file.jsonl
                    model_match = _re.search(r'/run\d+/([^/]+)/', file_path)
                    model_slug = model_match.group(1) if model_match else "unknown"
                    clean_metrics = {k: float(round(float(v), 6)) for k, v in metrics_dict.items()
                                    if k in metric_keys and v is not None}
                    per_run_metrics.append({
                        "run": run_num,
                        "datasetFile": Path(file_path).name,
                        "metrics": clean_metrics,
                    })
                    # Collect per-model data
                    if model_slug not in _h_per_model:
                        _h_per_model[model_slug] = {"metrics_list": [], "per_run": []}
                    _h_per_model[model_slug]["metrics_list"].append(clean_metrics)
                    _h_per_model[model_slug]["per_run"].append({
                        "run": run_num,
                        "metrics": clean_metrics,
                    })

                # Build per-model entries
                h_per_model_entries = []
                for m_slug, m_data in _h_per_model.items():
                    m_avg = {}
                    for key in metric_keys:
                        values = [m.get(key) for m in m_data["metrics_list"] if m.get(key) is not None]
                        if values:
                            m_avg[key] = float(round(sum(values) / len(values), 6))
                    h_per_model_entries.append({
                        "model": m_slug,
                        "modelSlug": m_slug,
                        "metrics": m_avg if m_avg else None,
                        "runs": sorted(m_data["per_run"], key=lambda x: x["run"]),
                    })
                _h_is_multi_model = len(h_per_model_entries) > 1

                # Aggregate metrics for this target (average across runs/models)
                all_metrics_dicts = [m for m, _ in target_eval_metrics]
                if all_metrics_dicts:
                    if len(all_metrics_dicts) > 1:
                        import numpy as np
                        avg_metrics = {}
                        avg_with_std = {}  # For Convex averageMetrics field
                        for key in metric_keys:
                            values = [m.get(key) for m in all_metrics_dicts if m.get(key) is not None]
                            if values:
                                try:
                                    values_array = np.array([float(v) for v in values])
                                    mean_val = float(round(np.mean(values_array), 6))
                                    std_val = float(round(np.std(values_array), 6))
                                    avg_metrics[key] = mean_val
                                    avg_metrics[f"{key}_std"] = std_val
                                    avg_with_std[key] = {"mean": mean_val, "std": std_val}
                                except (TypeError, ValueError):
                                    avg_metrics[key] = values[0]
                                    avg_with_std[key] = {"mean": float(round(float(values[0]), 6))}
                        _all_target_metrics.append(avg_metrics)
                    else:
                        _all_target_metrics.append(all_metrics_dicts[0])
                        avg_with_std = {k: {"mean": float(round(float(v), 6))}
                                        for k, v in all_metrics_dicts[0].items()
                                        if k in metric_keys and v is not None}

                    # Save consolidated metrics in CERA format (runs array + average + std)
                    import csv as _csv_module_h
                    import numpy as _np_h

                    _h_metric_keys = ["bleu", "rouge_l", "bertscore", "moverscore", "distinct_1", "distinct_2", "self_bleu"]
                    _h_metric_categories = {
                        "lexical": ["bleu", "rouge_l"],
                        "semantic": ["bertscore", "moverscore"],
                        "diversity": ["distinct_1", "distinct_2", "self_bleu"],
                    }
                    _h_metric_descriptions = {
                        "bleu": "N-gram overlap precision score",
                        "rouge_l": "Longest common subsequence recall",
                        "bertscore": "Contextual similarity using BERT embeddings",
                        "moverscore": "Earth mover distance on word embeddings",
                        "distinct_1": "Unique unigram ratio",
                        "distinct_2": "Unique bigram ratio",
                        "self_bleu": "Intra-corpus similarity (lower = more diverse)",
                    }

                    # Group by run number and average across models within each run
                    _h_per_run = {}
                    for prm in per_run_metrics:
                        rn = prm["run"]
                        if rn not in _h_per_run:
                            _h_per_run[rn] = []
                        _h_per_run[rn].append(prm["metrics"])

                    h_all_run_metrics = []
                    for rn in sorted(_h_per_run.keys()):
                        run_metrics_list = _h_per_run[rn]
                        run_avg = {}
                        for key in _h_metric_keys:
                            vals = [m.get(key) for m in run_metrics_list if m.get(key) is not None]
                            if vals:
                                run_avg[key] = float(round(sum(vals) / len(vals), 6))
                        h_all_run_metrics.append(run_avg)

                    # Compute average/std across runs
                    h_avg_metrics = {}
                    h_std_metrics = {}
                    for key in _h_metric_keys:
                        vals = [m.get(key) for m in h_all_run_metrics if m.get(key) is not None]
                        if vals:
                            arr = _np_h.array([float(v) for v in vals])
                            h_avg_metrics[key] = float(round(_np_h.mean(arr), 6))
                            h_std_metrics[key] = float(round(_np_h.std(arr), 6)) if len(vals) > 1 else 0.0

                    h_metrics_dir = Path(target_job_paths["metrics"])
                    h_metrics_dir.mkdir(parents=True, exist_ok=True)

                    # Save CSV per category (matching CERA format)
                    for category, cat_metrics in _h_metric_categories.items():
                        csv_path = h_metrics_dir / f"{category}-metrics.csv"
                        with open(csv_path, "w", newline="", encoding="utf-8") as f:
                            writer = _csv_module_h.writer(f)
                            if _h_is_multi_model:
                                writer.writerow(["run", "model", "metric", "score", "description"])
                                for prm in per_run_metrics:
                                    model_match = None
                                    for _, fp in target_eval_metrics:
                                        if f"/run{prm['run']}/" in fp:
                                            import re as _re2
                                            mm = _re2.search(r'/run\d+/([^/]+)/', fp)
                                            if mm:
                                                model_match = mm.group(1)
                                            break
                                    m_slug = model_match or "unknown"
                                    for metric in cat_metrics:
                                        val = prm["metrics"].get(metric)
                                        if val is not None:
                                            writer.writerow([prm["run"], m_slug, metric, f"{val:.6f}", _h_metric_descriptions.get(metric, "")])
                                for run_idx, run_met in enumerate(h_all_run_metrics, 1):
                                    for metric in cat_metrics:
                                        val = run_met.get(metric)
                                        if val is not None:
                                            writer.writerow([run_idx, "ALL", metric, f"{val:.6f}", f"Average across models"])
                                for metric in cat_metrics:
                                    val = h_avg_metrics.get(metric)
                                    if val is not None:
                                        writer.writerow(["avg", "ALL", metric, f"{val:.6f}", f"Average across {len(h_all_run_metrics)} runs"])
                                for metric in cat_metrics:
                                    val = h_std_metrics.get(metric)
                                    if val is not None:
                                        writer.writerow(["std", "ALL", metric, f"{val:.6f}", "Standard deviation"])
                            else:
                                writer.writerow(["run", "metric", "score", "description"])
                                for run_idx, run_met in enumerate(h_all_run_metrics, 1):
                                    for metric in cat_metrics:
                                        val = run_met.get(metric)
                                        if val is not None:
                                            writer.writerow([run_idx, metric, f"{val:.6f}", _h_metric_descriptions.get(metric, "")])
                                for metric in cat_metrics:
                                    val = h_avg_metrics.get(metric)
                                    if val is not None:
                                        writer.writerow(["avg", metric, f"{val:.6f}", f"Average across {len(h_all_run_metrics)} runs"])
                                for metric in cat_metrics:
                                    val = h_std_metrics.get(metric)
                                    if val is not None:
                                        writer.writerow(["std", metric, f"{val:.6f}", "Standard deviation"])

                    # Save consolidated JSON (CERA format with runs array)
                    all_runs_json = {
                        "runs": [],
                        "average": {},
                        "std": {},
                        "totalRuns": len(h_all_run_metrics),
                    }
                    for run_idx, run_met in enumerate(h_all_run_metrics, 1):
                        run_data = {"run": run_idx}
                        for category, cat_metrics in _h_metric_categories.items():
                            run_data[category] = {m: run_met.get(m) for m in cat_metrics if run_met.get(m) is not None}
                        all_runs_json["runs"].append(run_data)

                    for category, cat_metrics in _h_metric_categories.items():
                        all_runs_json["average"][category] = {m: h_avg_metrics.get(m) for m in cat_metrics if h_avg_metrics.get(m) is not None}
                        all_runs_json["std"][category] = {m: h_std_metrics.get(m) for m in cat_metrics}

                    all_runs_json["average"]["_flat"] = h_avg_metrics
                    all_runs_json["std"]["_flat"] = h_std_metrics

                    if _h_is_multi_model:
                        all_runs_json["totalModels"] = len(h_per_model_entries)
                        all_runs_json["perModel"] = h_per_model_entries

                    json_path = h_metrics_dir / "mdqa-results.json"
                    with open(json_path, "w") as f:
                        json.dump(all_runs_json, f, indent=2)

                    # Also save legacy mdqa_metrics_average.json for backward compatibility
                    agg_data = dict(_all_target_metrics[-1])
                    if _h_is_multi_model:
                        agg_data["totalModels"] = len(h_per_model_entries)
                        agg_data["perModel"] = h_per_model_entries
                    agg_path = h_metrics_dir / "mdqa_metrics_average.json"
                    with open(agg_path, "w") as f:
                        json.dump(agg_data, f, indent=2)

                    # Send per-target metrics to Convex
                    if convex:
                        try:
                            flat_metrics = {k: float(round(float(v), 6))
                                            for k, v in _all_target_metrics[-1].items()
                                            if k in metric_keys and v is not None}
                            save_args = {
                                "jobId": job_id,
                                "targetIndex": target_idx,
                                "targetLabel": f"{target.targetValue} {target.targetMode}",
                                "targetValue": target.targetValue,
                                "countMode": target.targetMode,
                                "metrics": flat_metrics,
                                "perRunMetrics": per_run_metrics if len(per_run_metrics) > 1 else None,
                                "averageMetrics": avg_with_std if len(all_metrics_dicts) > 1 else None,
                            }
                            # Add per-model metrics for Convex (strip runs detail)
                            if _h_is_multi_model:
                                save_args["perModelMetrics"] = [
                                    {"model": pm["model"], "modelSlug": pm["modelSlug"], "metrics": pm["metrics"]}
                                    for pm in h_per_model_entries
                                ]
                            await convex.run_mutation("jobs:saveTargetMetrics", save_args)
                        except Exception as e:
                            print(f"[Heuristic] Warning: Could not store per-target metrics: {e}")

                if convex:
                    await convex.add_log(job_id, "INFO", "MDQA",
                        f"[Target {target.targetValue}] Evaluation complete ({len(target_eval_metrics)} dataset(s))")

        # Execute targets (parallel or sequential based on parallelTargets flag)
        if heuristic_config.parallelTargets and len(effective_targets) > 1:
            await asyncio.gather(*[
                _run_heuristic_target(idx, t) for idx, t in enumerate(effective_targets)
            ])
        else:
            for idx, t in enumerate(effective_targets):
                await _run_heuristic_target(idx, t)

        # Report total reviews generated (before metrics — sets generatedCount)
        if convex:
            total_reviews = _mt_progress["reviews_collected"]
            if total_reviews > 0:
                try:
                    await convex.run_mutation("jobs:completeHeuristicGeneration", {
                        "jobId": job_id,
                        "reviewsCollected": total_reviews,
                    })
                except Exception as e:
                    print(f"[Heuristic] Warning: Could not report review count: {e}")

        # Send aggregated metrics to Convex
        if convex and _all_target_metrics:
            # If multiple targets, average across targets; if single target, use it directly
            if len(_all_target_metrics) > 1:
                import numpy as np
                combined = {}
                metric_keys = ["bleu", "rouge_l", "bertscore", "moverscore", "distinct_1", "distinct_2", "self_bleu"]
                for key in metric_keys:
                    values = [m.get(key) for m in _all_target_metrics if m.get(key) is not None]
                    if values:
                        try:
                            values_array = np.array([float(v) for v in values])
                            combined[key] = float(round(np.mean(values_array), 6))
                            combined[f"{key}_std"] = float(round(np.std(values_array), 6))
                        except (TypeError, ValueError):
                            combined[key] = values[0]
                final_metrics = combined
            else:
                # Single target — send its metrics directly
                final_metrics = {}
                metric_keys = ["bleu", "rouge_l", "bertscore", "moverscore", "distinct_1", "distinct_2", "self_bleu"]
                for k, v in _all_target_metrics[0].items():
                    if v is not None and (k in metric_keys or k.endswith("_std")):
                        try:
                            final_metrics[k] = float(round(float(v), 6))
                        except (TypeError, ValueError):
                            pass

            if final_metrics:
                try:
                    await convex.run_mutation("jobs:setEvaluationMetrics", {
                        "jobId": job_id,
                        "evaluationMetrics": final_metrics,
                    })
                except Exception as e:
                    print(f"[Heuristic] Warning: Could not store eval metrics: {e}")

        # Save tokens before completion
        _save_tokens_incremental()

        # Complete job
        if convex:
            await convex.run_mutation("jobs:complete", {"jobId": job_id})
            await convex.add_log(job_id, "INFO", "Heuristic",
                f"Pipeline complete! ({len(effective_targets)} targets × {len(effective_models)} models, "
                f"{_mt_progress['reviews_collected']} reviews)")

    except Exception as e:
        import traceback
        error_msg = f"Heuristic pipeline failed: {str(e)}"
        print(f"[Heuristic] Error: {error_msg}")
        print(traceback.format_exc())
        # Save whatever tokens we have on failure
        try:
            _save_tokens_incremental()
        except Exception:
            pass
        if convex:
            await convex.run_mutation("jobs:setFailed", {
                "jobId": job_id,
                "error": error_msg,
            })


def subsample_by_greedy_accumulation(
    reviews: list[dict],
    target_sentences: int,
    count_mode: str = "sentences",
    seed: int = 42,
) -> list[dict]:
    """
    Subsample reviews using greedy review accumulation.

    Shuffles reviews randomly, picks whole reviews one by one until sentence
    count >= target. Slight overshoot is expected and OK.

    Args:
        reviews: List of review dicts with 'sentences' field
        target_sentences: Target sentence count (or review count if count_mode='reviews')
        count_mode: "sentences" or "reviews"
        seed: Random seed for reproducibility
    Returns:
        List of sampled review dicts (preserving original structure)
    """
    import copy
    import random

    rng = random.Random(seed)
    shuffled = list(range(len(reviews)))
    rng.shuffle(shuffled)

    if count_mode == "reviews":
        selected_indices = shuffled[:target_sentences]
        return [copy.deepcopy(reviews[i]) for i in selected_indices]

    # Sentence mode: accumulate whole reviews until sentence count >= target
    sampled = []
    total = 0
    for idx in shuffled:
        review = reviews[idx]
        sentences = review.get("sentences", [])
        n = len(sentences) if sentences else 1
        sampled.append(copy.deepcopy(review))
        total += n
        if total >= target_sentences:
            break

    return sampled


def escape_xml(text: str) -> str:
    """Escape special XML characters."""
    return (text
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;"))


async def _execute_real_dataset_pipeline(
    job_id: str,
    job_name: str,
    dataset_file: Optional[str],
    evaluation_config: Optional[dict],
    jobs_directory: str,
    convex_url: Optional[str],
    convex_token: Optional[str],
):
    """
    Execute real dataset evaluation pipeline with multi-target subsampling.

    For each target size:
    1. Subsample source dataset using greedy review accumulation (whole reviews)
    2. Write subsample as both JSONL and SemEval XML: {job-name}-{explicit|implicit}.{jsonl|xml}
    3. Run execute_evaluation (which handles self-test + MDQA internally)
    4. Results written to datasets/{size}/metrics/mdqa-results.json
    """
    from pathlib import Path
    import json
    import xml.etree.ElementTree as ET

    convex = None
    if convex_url and convex_token:
        convex = ConvexClient(convex_url, convex_token, pocketbase_url=os.environ.get("POCKETBASE_URL"))

    try:
        # Create job directory
        job_paths = create_job_directory(jobs_directory, job_id, job_name)

        # Extract targets from evaluationConfig
        targets = []
        if evaluation_config and "targets" in evaluation_config:
            for t in evaluation_config["targets"]:
                targets.append({
                    "count_mode": t.get("count_mode", "sentences"),
                    "target_value": t.get("target_value", 100),
                })

        if not targets:
            raise ValueError("No target sizes specified for real dataset pipeline")

        if convex:
            target_labels = [f"{t['target_value']} {t['count_mode']}" for t in targets]
            await convex.add_log(job_id, "INFO", "Pipeline",
                f"Starting real dataset pipeline with {len(targets)} targets: {', '.join(target_labels)}")
            await convex.update_progress(job_id, 5, "Loading source dataset")

        # Locate the uploaded source dataset file
        dataset_dir = Path(job_paths["dataset"])
        source_path = None
        if dataset_file:
            candidate = dataset_dir / dataset_file
            if candidate.exists():
                source_path = candidate
        if not source_path:
            # Auto-discover
            for ext in [".jsonl", ".csv", ".xml"]:
                candidates = list(dataset_dir.glob(f"*{ext}"))
                if candidates:
                    source_path = candidates[0]
                    break
        if not source_path or not source_path.exists():
            raise FileNotFoundError(f"Source dataset not found in {dataset_dir}")

        # Parse source dataset into review dicts
        source_reviews: list[dict] = []
        if str(source_path).endswith(".jsonl"):
            with open(source_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        source_reviews.append(json.loads(line))
        elif str(source_path).endswith(".xml"):
            tree = ET.parse(source_path)
            root = tree.getroot()
            for review_elem in root.findall("./Review"):
                review_sentences = []
                sentences_wrapper = review_elem.find("./sentences")
                sentence_elems = (
                    sentences_wrapper.findall("./sentence") if sentences_wrapper is not None
                    else review_elem.findall("./sentence")
                )
                for sent_elem in sentence_elems:
                    text_elem = sent_elem.find("text")
                    if text_elem is not None and text_elem.text:
                        sent_dict: dict = {"text": text_elem.text.strip()}
                        # Preserve opinion annotations if present
                        opinions_elem = sent_elem.find("Opinions")
                        if opinions_elem is not None:
                            opinions = []
                            for op_elem in opinions_elem.findall("Opinion"):
                                opinions.append({
                                    "target": op_elem.get("target", "NULL"),
                                    "category": op_elem.get("category", ""),
                                    "polarity": op_elem.get("polarity", ""),
                                    "from": op_elem.get("from", "0"),
                                    "to": op_elem.get("to", "0"),
                                })
                            sent_dict["opinions"] = opinions
                        review_sentences.append(sent_dict)
                if review_sentences:
                    source_reviews.append({"sentences": review_sentences})
        elif str(source_path).endswith(".csv"):
            import csv as csv_mod
            with open(source_path, encoding="utf-8") as f:
                reader = csv_mod.DictReader(f)
                source_reviews = list(reader)
        else:
            # Fallback: try JSONL
            with open(source_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        source_reviews.append(json.loads(line))

        if not source_reviews:
            raise ValueError(f"No reviews found in source dataset: {source_path.name}")

        # Detect explicit vs implicit: explicit if any opinion has a real target (not "NULL")
        is_explicit = False
        for review in source_reviews:
            for sent in review.get("sentences", []):
                for op in sent.get("opinions", []):
                    if op.get("target", "NULL") not in ("NULL", "null", ""):
                        is_explicit = True
                        break
                if is_explicit:
                    break
            if is_explicit:
                break
        dataset_mode = "explicit" if is_explicit else "implicit"

        # Sanitize job name for filenames
        import re as _re
        safe_job_name = _re.sub(r'[^\w\-]', '-', job_name).strip('-').lower()
        safe_job_name = _re.sub(r'-+', '-', safe_job_name)

        total_sentences = sum(len(r.get("sentences", [])) for r in source_reviews)
        print(f"[Real Pipeline] Loaded {len(source_reviews)} reviews ({total_sentences} sentences) from {source_path.name}")
        if convex:
            await convex.add_log(job_id, "INFO", "Pipeline",
                f"Loaded {len(source_reviews)} reviews ({total_sentences} sentences) from {source_path.name}")

        # Process each target
        for idx, target in enumerate(targets):
            target_value = target["target_value"]
            count_mode = target["count_mode"]
            progress_base = 10 + int((idx / len(targets)) * 80)

            if convex:
                await convex.update_progress(job_id, progress_base, f"Target {target_value}: subsampling")
                await convex.add_log(job_id, "INFO", "Pipeline",
                    f"[Target {idx+1}/{len(targets)}] Subsampling {target_value} {count_mode}")

            # Subsample using greedy review accumulation
            sampled_reviews = subsample_by_greedy_accumulation(
                reviews=source_reviews,
                target_sentences=target_value,
                count_mode=count_mode,
                seed=42 + idx,  # Different seed per target for independence
            )

            sampled_sentences = sum(len(r.get("sentences", [])) for r in sampled_reviews)
            print(f"[Real Pipeline] Target {target_value}: sampled {len(sampled_reviews)} reviews ({sampled_sentences} sentences)")
            if convex:
                await convex.add_log(job_id, "INFO", "Pipeline",
                    f"Sampled {len(sampled_reviews)} reviews ({sampled_sentences} sentences)")

            # Create target directory structure
            target_dir = Path(job_paths["datasets"]) / str(target_value)
            target_dir.mkdir(parents=True, exist_ok=True)
            metrics_dir = target_dir / "metrics"
            metrics_dir.mkdir(exist_ok=True)

            # Write subsample in both JSONL and SemEval XML
            file_base = f"{safe_job_name}-{dataset_mode}"

            # JSONL
            jsonl_file = target_dir / f"{file_base}.jsonl"
            with open(jsonl_file, "w", encoding="utf-8") as f:
                for review in sampled_reviews:
                    f.write(json.dumps(review, ensure_ascii=False) + "\n")

            # SemEval XML
            xml_file = target_dir / f"{file_base}.xml"
            xml_content = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
            xml_content += '<Reviews>\n'
            for r_idx, review in enumerate(sampled_reviews):
                xml_content += f'  <Review rid="{r_idx}">\n'
                xml_content += '    <sentences>\n'
                for s_idx, sent in enumerate(review.get("sentences", [])):
                    text_escaped = sent["text"].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
                    xml_content += f'      <sentence id="{r_idx}:{s_idx}">\n'
                    xml_content += f'        <text>{text_escaped}</text>\n'
                    opinions = sent.get("opinions", [])
                    if opinions:
                        xml_content += '        <Opinions>\n'
                        for op in opinions:
                            target_escaped = str(op.get("target", "NULL")).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
                            xml_content += f'          <Opinion target="{target_escaped}" category="{op.get("category", "")}" polarity="{op.get("polarity", "")}" from="{op.get("from", "0")}" to="{op.get("to", "0")}" />\n'
                        xml_content += '        </Opinions>\n'
                    xml_content += '      </sentence>\n'
                xml_content += '    </sentences>\n'
                xml_content += '  </Review>\n'
            xml_content += '</Reviews>\n'
            with open(xml_file, "w", encoding="utf-8") as f:
                f.write(xml_content)

            # Build target-specific job_paths for execute_evaluation
            target_job_paths = dict(job_paths)
            target_job_paths["dataset"] = str(target_dir)
            target_job_paths["metrics"] = str(metrics_dir)

            # Run MDQA evaluation (handles self-test internally)
            if convex:
                await convex.update_progress(job_id, progress_base + 5, f"Target {target_value}: evaluating")
                await convex.add_log(job_id, "INFO", "MDQA",
                    f"[Target {idx+1}/{len(targets)}] Starting evaluation for {target_value} {count_mode}")

            await execute_evaluation(
                job_id=job_id,
                job_paths=target_job_paths,
                evaluation_config=evaluation_config,
                reviews_data=sampled_reviews,
                save_files=True,
                convex=convex,
            )

            if convex:
                await convex.add_log(job_id, "INFO", "MDQA",
                    f"[Target {idx+1}/{len(targets)}] Evaluation complete for {target_value}")

        # Mark job as completed
        if convex:
            await convex.update_progress(job_id, 100, "Complete")
            await convex.complete_job(job_id)
            await convex.add_log(job_id, "INFO", "Pipeline",
                f"Real dataset pipeline complete — {len(targets)} targets evaluated")

        print(f"[Real Pipeline] Pipeline complete for job {job_id}")

    except Exception as e:
        print(f"[Real Pipeline] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        if convex:
            await convex.fail_job(job_id, str(e))
            await convex.add_log(job_id, "ERROR", "Pipeline", f"Pipeline failed: {e}")


async def execute_pipeline(
    job_id: str,
    job_name: str,
    config: Optional[JobConfig],  # Optional for heuristic method
    phases: list[str],
    api_key: str,
    tavily_api_key: Optional[str],
    jobs_directory: str,
    convex_url: Optional[str],
    convex_token: Optional[str],
    evaluation_config: Optional[dict],
    dataset_file: Optional[str],
    reused_from_job_dir: Optional[str] = None,
    reference_dataset: Optional[dict] = None,
    method: str = "cera",
    heuristic_config: Optional[HeuristicConfig] = None,
    rde_usage: Optional[list[dict]] = None,
):
    """Execute selected pipeline phases sequentially."""
    from pathlib import Path

    convex = None
    if convex_url and convex_token:
        convex = ConvexClient(convex_url, convex_token, pocketbase_url=os.environ.get("POCKETBASE_URL"))

    # ========================================
    # HEURISTIC METHOD (RQ1 Baseline)
    # ========================================
    if method == "heuristic" and heuristic_config:
        await execute_heuristic_pipeline(
            job_id=job_id,
            job_name=job_name,
            heuristic_config=heuristic_config,
            api_key=api_key,
            jobs_directory=jobs_directory,
            convex_url=convex_url,
            convex_token=convex_token,
            evaluation_config=evaluation_config,
            rde_usage=rde_usage,
        )
        return  # Heuristic pipeline is complete

    # ========================================
    # REAL DATASET METHOD (RQ1 Real Baseline)
    # Subsample source dataset at each target size, then self-test + MDQA.
    # ========================================
    if method == "real" and "evaluation" in phases and len(phases) == 1:
        await _execute_real_dataset_pipeline(
            job_id=job_id,
            job_name=job_name,
            dataset_file=dataset_file,
            evaluation_config=evaluation_config,
            jobs_directory=jobs_directory,
            convex_url=convex_url,
            convex_token=convex_token,
        )
        return  # Real dataset pipeline is complete

    try:
        # Get or create job directory
        job_paths = create_job_directory(jobs_directory, job_id, job_name)

        # Initialize token usage tracker
        from cera.llm.usage import UsageTracker
        usage_tracker = UsageTracker()
        tokens_json_path = Path(job_paths["reports"]) / "tokens.json"

        # Import pre-job RDE usage records if available
        if rde_usage:
            usage_tracker.import_records(rde_usage)
            print(f"[Tokens] Imported {len(rde_usage)} RDE usage records")

        def _save_tokens_incremental():
            """Incrementally save tokens.json (safe to call anytime)."""
            try:
                if usage_tracker.total_tokens > 0:
                    _method = method if method != "heuristic" else "cera"
                    usage_tracker.save_tokens_json(tokens_json_path, job_id, _method)
            except Exception as _e:
                print(f"[Tokens] Warning: Failed to save tokens.json: {_e}")

        # ========================================
        # Handle Composition Reuse (copy context files from source job)
        # ========================================
        if reused_from_job_dir and "composition" not in phases:
            import shutil
            source_contexts_dir = Path(reused_from_job_dir) / "contexts"
            target_contexts_dir = Path(job_paths["contexts"])

            if source_contexts_dir.exists():
                # Copy all context files from source job
                for context_file in source_contexts_dir.glob("*.json"):
                    shutil.copy2(context_file, target_contexts_dir / context_file.name)

                if convex:
                    await convex.add_log(job_id, "INFO", "Pipeline", f"Copied composition contexts from: {reused_from_job_dir}")
            else:
                if convex:
                    await convex.add_log(job_id, "WARN", "Pipeline", f"Source job contexts not found: {source_contexts_dir}")

            # Copy reviewer personas from source job
            source_personas_dir = Path(reused_from_job_dir) / "reviewer-personas"
            target_personas_dir = Path(job_paths["root"]) / "reviewer-personas"
            if source_personas_dir.exists() and any(source_personas_dir.iterdir()):
                target_personas_dir.mkdir(parents=True, exist_ok=True)
                for persona_file in source_personas_dir.glob("*.md"):
                    shutil.copy2(persona_file, target_personas_dir / persona_file.name)
                if convex:
                    count = len(list(target_personas_dir.glob("*.md")))
                    await convex.add_log(job_id, "INFO", "Pipeline", f"Copied {count} reviewer personas from source job")

            # Copy reference dataset files from source job (for MDQA evaluation)
            # Check both datasets/ (new) and dataset/ (legacy) in source job
            _src_ref_dirs = [Path(reused_from_job_dir) / "datasets", Path(reused_from_job_dir) / "dataset"]
            source_dataset_dir = next((d for d in _src_ref_dirs if d.exists() and list(d.glob("reference_*"))), None)
            target_dataset_dir = Path(job_paths["datasets"])
            if source_dataset_dir:
                for ref_file in source_dataset_dir.glob("reference_*"):
                    shutil.copy2(ref_file, target_dataset_dir / ref_file.name)
                    if convex:
                        await convex.add_log(job_id, "INFO", "Pipeline", f"Copied reference dataset: {ref_file.name}")

        # ========================================
        # Save config.json for this job
        # ========================================
        config_path = save_job_config(
            job_paths=job_paths,
            job_name=job_name,
            config=config,
            phases=phases,
            evaluation_config=evaluation_config,
            reused_from=reused_from_job_dir,
            reference_dataset=reference_dataset,
        )

        if convex:
            # Store config path in Convex
            await convex.run_mutation("jobs:setConfigPath", {
                "jobId": job_id,
                "configPath": config_path,
            })
            await convex.add_log(job_id, "INFO", "Pipeline", f"Starting pipeline with phases: {', '.join(phases)}")
            if convex.has_fast_updates:
                await convex.add_log(job_id, "INFO", "Pipeline", "[PB] PocketBase real-time mode active (progress + logs)")

            # Early GPU/CPU detection - set immediately so UI shows the badge
            try:
                import torch
                if torch.cuda.is_available():
                    device_name = torch.cuda.get_device_name(0)
                    await convex.set_evaluation_device(job_id, "GPU", device_name)
                    await convex.add_log(job_id, "INFO", "Pipeline", f"GPU detected: {device_name}")
                else:
                    await convex.set_evaluation_device(job_id, "CPU")
                    await convex.add_log(job_id, "INFO", "Pipeline", "Running on CPU (no CUDA GPU available)")
            except ImportError:
                await convex.set_evaluation_device(job_id, "CPU")
                await convex.add_log(job_id, "INFO", "Pipeline", "Running on CPU (PyTorch not available)")

        # ========================================
        # Phase 1: COMPOSITION (SIL + MAV)
        # ========================================
        if "composition" in phases:
            if convex:
                await convex.start_composing(job_id, job_paths["root"])
                await convex.add_log(job_id, "INFO", "SIL", "Starting composition phase (SIL/MAV)...")

            # Run full composition (SIL + MAV + RGM + ACM)
            result = await execute_composition_simple(
                job_id=job_id,
                job_name=job_name,
                config=config,
                api_key=api_key,
                tavily_api_key=tavily_api_key,
                jobs_directory=jobs_directory,
                convex_url=convex_url,
                convex_token=convex_token,
                usage_tracker=usage_tracker,
            )

            if convex:
                # Store contexts in Convex and mark as composed
                await convex.run_mutation("jobs:completeComposition", {
                    "jobId": job_id,
                    "subjectContext": result.get("subjectContext"),
                    "reviewerContext": result.get("reviewerContext"),
                    "attributesContext": result.get("attributesContext"),
                })
                await convex.add_log(job_id, "INFO", "SIL", "Composition phase complete.")

            # Update config.json with composition metadata
            try:
                config_json_path = Path(job_paths["root"]) / "config.json"
                if config_json_path.exists():
                    with open(config_json_path, "r") as f:
                        config_data = json.load(f)
                    # Count personas
                    personas_dir = Path(job_paths["root"]) / "reviewer-personas"
                    persona_count = len(list(personas_dir.glob("*.md"))) if personas_dir.exists() else 0
                    # Count structure templates
                    structures_path = Path(job_paths["contexts"]) / "structure-variants.json"
                    structure_count = 0
                    if structures_path.exists():
                        with open(structures_path) as f:
                            structure_count = len(json.load(f))
                    config_data["composition_meta"] = {
                        "persona_count": persona_count,
                        "structure_template_count": structure_count,
                    }
                    with open(config_json_path, "w", encoding="utf-8") as f:
                        json.dump(config_data, f, indent=2, default=str)
            except Exception as e:
                print(f"[Pipeline] Warning: Could not update config.json with composition metadata: {e}", flush=True)

            # Save tokens after composition phase
            _save_tokens_incremental()

        # ========================================
        # Target-based Pipeline (all jobs use datasets/{size}/ structure)
        # Runs generation + evaluation per target, then returns early.
        # ========================================
        effective_targets = config.generation.get_effective_targets() if config and config.generation else []

        # For eval-only reruns without config: discover targets from datasets/ directory
        if not effective_targets and "evaluation" in phases:
            from pathlib import Path as _DiscPath
            datasets_dir = _DiscPath(job_paths.get("datasets", ""))
            if datasets_dir.exists():
                for child in sorted(datasets_dir.iterdir()):
                    if child.is_dir() and child.name.isdigit():
                        # Discover runs and models from directory structure
                        run_dirs = sorted([d for d in child.iterdir() if d.is_dir() and d.name.startswith("run")])
                        num_runs = len(run_dirs) if run_dirs else 1
                        effective_targets.append(TargetDataset(
                            count_mode="sentences",  # default assumption
                            target_value=int(child.name),
                            total_runs=num_runs,
                        ))
                if effective_targets:
                    print(f"[Pipeline] Discovered {len(effective_targets)} targets from datasets/ directory: "
                          f"{[t.target_value for t in effective_targets]}")

        if len(effective_targets) >= 1 and ("generation" in phases or "evaluation" in phases):
            import asyncio as _mt_asyncio

            # Resolve models from config or discover from directory structure
            if config and config.generation:
                all_models = config.generation.models or [config.generation.model]
                all_models = [m for m in all_models if m]
            else:
                # Discover models from first target's run directory
                all_models = []
                from pathlib import Path as _DiscPath2
                datasets_dir = _DiscPath2(job_paths.get("datasets", ""))
                first_target = datasets_dir / str(effective_targets[0].target_value) / "run1"
                if first_target.exists():
                    _skip_dirs = {"amls", "personas", "reviewer-personas", "metrics", "__pycache__"}
                    for child in sorted(first_target.iterdir()):
                        if child.is_dir() and child.name not in _skip_dirs:
                            all_models.append(child.name)
                if not all_models:
                    all_models = ["unknown"]
                print(f"[Pipeline] Discovered models from directory: {all_models}")
            is_multi_model = len(all_models) > 1
            parallel_models = getattr(config.generation, 'parallel_models', True) if config and config.generation else True

            if convex:
                target_sizes = [f"{t.target_value} {t.count_mode}" for t in effective_targets]
                await convex.add_log(job_id, "INFO", "Pipeline",
                    f"Multi-target pipeline: {len(effective_targets)} targets ({', '.join(target_sizes)})")
                if "generation" in phases:
                    await convex.run_mutation("jobs:startGeneration", {"id": job_id})

            # GPU evaluation lock (shared across all targets for sequential GPU access)
            eval_lock = _mt_asyncio.Lock()

            async def _run_target(target_idx: int, target: TargetDataset):
                """Run generation + evaluation for one target size."""
                target_label = f"{target.target_value} {target.count_mode}"

                if convex:
                    await convex.add_log(job_id, "INFO", "Pipeline",
                        f"[Target {target_idx+1}/{len(effective_targets)}] Starting: {target_label}")
                    await convex.run_mutation("jobs:updateTargetProgress", {
                        "jobId": job_id,
                        "targetIndex": target_idx,
                        "targetLabel": target_label,
                        "status": "generating",
                        "progress": 0,
                    })

                # Create target directory structure
                target_paths = create_target_directory(
                    job_paths["datasets"], target.target_value, target.count_mode,
                    target.total_runs, all_models,
                )

                # Modified job_paths for evaluation (metrics dir under target)
                target_job_paths = dict(job_paths)
                target_job_paths["dataset"] = target_paths["target"]
                target_job_paths["metrics"] = target_paths["metrics"]

                # Create config copy with target-specific generation fields (only needed for generation)
                target_config = None
                if config and config.generation:
                    target_config = config.model_copy(deep=True)
                    target_config.generation.count_mode = target.count_mode
                    if target.count_mode == "sentences":
                        target_config.generation.target_sentences = target.target_value
                        target_config.generation.count = max(1, target.target_value // 5)
                    else:
                        target_config.generation.count = target.target_value
                        target_config.generation.target_sentences = None
                    target_config.generation.batch_size = target.batch_size
                    target_config.generation.request_size = target.request_size
                    target_config.generation.total_runs = target.total_runs
                    target_config.generation.neb_enabled = target.neb_depth > 0
                    target_config.generation.neb_depth = target.neb_depth

                total_target_runs = target.total_runs
                parallel_runs = target.runs_mode == "parallel"

                # --- Generation ---
                if "generation" in phases:
                    async def _gen_model_run(model_id: str, current_run: int):
                        """Generate for one model + one run."""
                        model_slug = model_id.split("/")[-1]
                        model_config = target_config.model_copy(deep=True)
                        model_config.generation.model = model_id

                        # Output directories for this run
                        run_paths = target_paths["runs"].get(current_run, {})
                        dataset_dir = run_paths.get("models", {}).get(model_id) or target_paths["target"]
                        amls_dir = run_paths.get("amls") or None

                        if convex:
                            run_label = f" run {current_run}/{total_target_runs}" if total_target_runs > 1 else ""
                            model_label = f" [{model_slug}]" if is_multi_model else ""
                            await convex.add_log(job_id, "INFO", "AML",
                                f"[Target {target.target_value}]{model_label}{run_label} Generating...")

                        gen_result = await execute_generation(
                            job_id=job_id,
                            job_dir=job_paths["root"],
                            config=model_config,
                            api_key=api_key,
                            convex_url=convex_url,
                            convex_token=convex_token,
                            should_complete_job=False,
                            model_tag=model_slug if is_multi_model else None,
                            current_run=current_run,
                            total_runs=total_target_runs,
                            dataset_dir_override=dataset_dir,
                            amls_dir_override=amls_dir,
                            usage_tracker=usage_tracker,
                        )

                        if convex:
                            await convex.add_log(job_id, "INFO", "AML",
                                f"[Target {target.target_value}]{model_label}{run_label} Generation complete")

                        return gen_result

                    # Execute generation: models x runs
                    gen_tasks = []
                    for model_id in all_models:
                        for current_run in range(1, total_target_runs + 1):
                            gen_tasks.append((model_id, current_run))

                    if parallel_runs and parallel_models:
                        # Fully parallel (all model/run combos at once)
                        await _mt_asyncio.gather(*[_gen_model_run(m, r) for m, r in gen_tasks])
                    elif parallel_models:
                        # Parallel models, sequential runs
                        for current_run in range(1, total_target_runs + 1):
                            await _mt_asyncio.gather(*[
                                _gen_model_run(m, current_run) for m in all_models
                            ])
                    elif parallel_runs:
                        # Sequential models, parallel runs
                        for model_id in all_models:
                            await _mt_asyncio.gather(*[
                                _gen_model_run(model_id, r) for r in range(1, total_target_runs + 1)
                            ])
                    else:
                        # Fully sequential
                        for m, r in gen_tasks:
                            await _gen_model_run(m, r)

                # Save tokens after this target's generation
                _save_tokens_incremental()

                # Update target progress: generation done
                if convex:
                    await convex.run_mutation("jobs:updateTargetProgress", {
                        "jobId": job_id,
                        "targetIndex": target_idx,
                        "targetLabel": target_label,
                        "status": "evaluating",
                        "progress": 50,
                    })

                # --- Evaluation ---
                if "evaluation" in phases:
                    if convex:
                        await convex.add_log(job_id, "INFO", "MDQA",
                            f"[Target {target.target_value}] Starting evaluation...")

                    import numpy as _np
                    import csv as _csv_module
                    from pathlib import Path as _Path
                    target_eval_metrics = []  # List of {"model": str, "run": int, "metrics": dict}

                    for model_id in all_models:
                        model_slug = model_id.split("/")[-1]

                        for current_run in range(1, total_target_runs + 1):
                            run_paths = target_paths["runs"].get(current_run, {})
                            model_dir = run_paths.get("models", {}).get(model_id)

                            if model_dir:
                                model_dir_path = _Path(model_dir)
                                # Find dataset file in model dir
                                dataset_file_path = None
                                for ext in [".jsonl", ".csv", ".xml"]:
                                    candidates = list(model_dir_path.glob(f"*{ext}"))
                                    if candidates:
                                        dataset_file_path = candidates[0]
                                        break

                                if dataset_file_path:
                                    async with eval_lock:
                                        eval_metrics = await execute_evaluation(
                                            job_id=job_id,
                                            job_paths=target_job_paths,
                                            evaluation_config=evaluation_config,
                                            dataset_file=str(dataset_file_path),
                                            convex=convex,
                                            run_number=current_run if total_target_runs > 1 else None,
                                            save_files=False,  # We save consolidated files below
                                        )
                                        if eval_metrics:
                                            target_eval_metrics.append({
                                                "model": model_id,
                                                "run": current_run,
                                                "metrics": eval_metrics,
                                            })

                    eval_count = len(target_eval_metrics)
                    if convex:
                        await convex.add_log(job_id, "INFO", "MDQA",
                            f"[Target {target.target_value}] Evaluation complete ({eval_count} evaluations)")

                    # --- Aggregate metrics across runs (with cross-run avg/std) ---
                    if target_eval_metrics:
                        _metric_keys = ["bleu", "rouge_l", "bertscore", "moverscore", "distinct_1", "distinct_2", "self_bleu"]
                        _metric_categories = {
                            "lexical": ["bleu", "rouge_l"],
                            "semantic": ["bertscore", "moverscore"],
                            "diversity": ["distinct_1", "distinct_2", "self_bleu"],
                        }
                        _metric_descriptions = {
                            "bleu": "N-gram overlap precision score",
                            "rouge_l": "Longest common subsequence recall",
                            "bertscore": "Contextual similarity using BERT embeddings",
                            "moverscore": "Earth mover distance on word embeddings",
                            "distinct_1": "Unique unigram ratio",
                            "distinct_2": "Unique bigram ratio",
                            "self_bleu": "Intra-corpus similarity (lower = more diverse)",
                        }

                        # Group by run: for each run, average across models to get per-run metrics
                        _per_run_metrics = {}  # run -> [metrics_dicts]
                        for em in target_eval_metrics:
                            run_num = em["run"]
                            if run_num not in _per_run_metrics:
                                _per_run_metrics[run_num] = []
                            _per_run_metrics[run_num].append(em["metrics"])

                        all_run_metrics = []  # Ordered list of per-run aggregated metrics
                        convex_per_run = []  # For Convex storage
                        for run_num in sorted(_per_run_metrics.keys()):
                            run_metrics_list = _per_run_metrics[run_num]
                            # Average across models within this run
                            run_avg = {}
                            for key in _metric_keys:
                                values = [m.get(key) for m in run_metrics_list if m.get(key) is not None]
                                if values:
                                    run_avg[key] = float(round(sum(values) / len(values), 6))
                            all_run_metrics.append(run_avg)
                            convex_per_run.append({
                                "run": run_num,
                                "datasetFile": f"run{run_num}",
                                "metrics": {k: run_avg.get(k) for k in _metric_keys if run_avg.get(k) is not None},
                            })

                        # Compute average/std across runs
                        avg_metrics = {}
                        std_metrics = {}
                        for key in _metric_keys:
                            values = [m.get(key) for m in all_run_metrics if m.get(key) is not None]
                            if values:
                                values_array = _np.array([float(v) for v in values])
                                avg_metrics[key] = float(round(_np.mean(values_array), 6))
                                std_metrics[key] = float(round(_np.std(values_array), 6)) if len(values) > 1 else 0.0

                        # --- Compute per-model breakdown (for disk AND Convex) ---
                        _per_model = {}  # model_id -> {metrics_list, model_slug, per_run}
                        for em in target_eval_metrics:
                            mid = em["model"]
                            if mid not in _per_model:
                                _per_model[mid] = {"metrics_list": [], "model_slug": mid.split("/")[-1], "per_run": []}
                            _per_model[mid]["metrics_list"].append(em.get("metrics", {}))
                            _per_model[mid]["per_run"].append({
                                "run": em["run"],
                                "metrics": {k: em["metrics"].get(k) for k in _metric_keys if em["metrics"].get(k) is not None},
                            })

                        per_model_entries = []
                        for mid, data in _per_model.items():
                            model_avg = {}
                            for key in _metric_keys:
                                values = [m.get(key) for m in data["metrics_list"] if m.get(key) is not None]
                                if values:
                                    model_avg[key] = float(round(sum(values) / len(values), 6))
                            per_model_entries.append({
                                "model": mid,
                                "modelSlug": data["model_slug"],
                                "metrics": model_avg if model_avg else None,
                                "runs": sorted(data["per_run"], key=lambda x: x["run"]),
                            })

                        _is_multi_model = len(per_model_entries) > 1

                        # --- Save consolidated files to target metrics dir ---
                        metrics_dir = _Path(target_paths["metrics"])
                        metrics_dir.mkdir(parents=True, exist_ok=True)

                        # Save CSV per category (long format with run column)
                        for category, metric_names in _metric_categories.items():
                            csv_path = metrics_dir / f"{category}-metrics.csv"
                            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                                writer = _csv_module.writer(f)
                                if _is_multi_model:
                                    writer.writerow(["run", "model", "metric", "score", "description"])
                                    # Per-model per-run rows
                                    for em in target_eval_metrics:
                                        run_num = em["run"]
                                        m_slug = em["model"].split("/")[-1]
                                        for metric in metric_names:
                                            val = em["metrics"].get(metric)
                                            if val is not None:
                                                writer.writerow([run_num, m_slug, metric, f"{val:.6f}", _metric_descriptions.get(metric, "")])
                                    # Cross-model averages per run
                                    for run_idx, run_metrics in enumerate(all_run_metrics, 1):
                                        for metric in metric_names:
                                            val = run_metrics.get(metric)
                                            if val is not None:
                                                writer.writerow([run_idx, "ALL", metric, f"{val:.6f}", f"Average across {len(per_model_entries)} models"])
                                    # Overall averages
                                    for metric in metric_names:
                                        val = avg_metrics.get(metric)
                                        if val is not None:
                                            writer.writerow(["avg", "ALL", metric, f"{val:.6f}", f"Average across {len(all_run_metrics)} runs"])
                                    # Per-model averages across runs
                                    for pm in per_model_entries:
                                        if pm["metrics"]:
                                            for metric in metric_names:
                                                val = pm["metrics"].get(metric)
                                                if val is not None:
                                                    writer.writerow(["avg", pm["modelSlug"], metric, f"{val:.6f}", f"Average across runs for {pm['modelSlug']}"])
                                    # Overall std
                                    for metric in metric_names:
                                        val = std_metrics.get(metric)
                                        if val is not None:
                                            writer.writerow(["std", "ALL", metric, f"{val:.6f}", "Standard deviation"])
                                else:
                                    writer.writerow(["run", "metric", "score", "description"])
                                    for run_idx, run_metrics in enumerate(all_run_metrics, 1):
                                        for metric in metric_names:
                                            val = run_metrics.get(metric)
                                            if val is not None:
                                                writer.writerow([run_idx, metric, f"{val:.6f}", _metric_descriptions.get(metric, "")])
                                    for metric in metric_names:
                                        val = avg_metrics.get(metric)
                                        if val is not None:
                                            writer.writerow(["avg", metric, f"{val:.6f}", f"Average across {len(all_run_metrics)} runs"])
                                    for metric in metric_names:
                                        val = std_metrics.get(metric)
                                        if val is not None:
                                            writer.writerow(["std", metric, f"{val:.6f}", "Standard deviation"])
                            print(f"[Evaluation] Saved {category} metrics: {csv_path}")

                        # Save consolidated JSON with all runs
                        all_runs_json = {
                            "runs": [],
                            "average": {},
                            "std": {},
                            "totalRuns": len(all_run_metrics),
                        }
                        for run_idx, run_metrics in enumerate(all_run_metrics, 1):
                            run_data = {"run": run_idx}
                            for category, metric_names in _metric_categories.items():
                                run_data[category] = {m: run_metrics.get(m) for m in metric_names if run_metrics.get(m) is not None}
                            all_runs_json["runs"].append(run_data)

                        for category, metric_names in _metric_categories.items():
                            all_runs_json["average"][category] = {m: avg_metrics.get(m) for m in metric_names if avg_metrics.get(m) is not None}
                            all_runs_json["std"][category] = {m: std_metrics.get(m) for m in metric_names}

                        all_runs_json["average"]["_flat"] = avg_metrics
                        all_runs_json["std"]["_flat"] = std_metrics

                        # Add per-model breakdown for multi-model jobs
                        if _is_multi_model:
                            all_runs_json["totalModels"] = len(per_model_entries)
                            all_runs_json["perModel"] = per_model_entries

                        json_path = metrics_dir / "mdqa-results.json"
                        with open(json_path, "w") as f:
                            json.dump(all_runs_json, f, indent=2)
                        print(f"[Evaluation] Saved consolidated metrics: {json_path}")

                        # --- Save to Convex ---
                        print(f"[Evaluation] Saving target {target_idx} metrics to Convex (convex={convex is not None})...", flush=True)
                        if convex:
                            try:
                                # Build averageMetrics for Convex (mean + std per metric)
                                convex_avg_metrics = {}
                                for key in _metric_keys:
                                    if key in avg_metrics:
                                        convex_avg_metrics[key] = {
                                            "mean": avg_metrics[key],
                                            "std": std_metrics.get(key, 0.0),
                                        }

                                # Overall flat metrics (average across all)
                                overall_metrics = {k: avg_metrics[k] for k in _metric_keys if k in avg_metrics}

                                # Strip per-run detail from per_model_entries for Convex
                                convex_per_model = [
                                    {"model": pm["model"], "modelSlug": pm["modelSlug"], "metrics": pm["metrics"]}
                                    for pm in per_model_entries
                                ]

                                print(f"[Evaluation] Calling saveTargetMetrics for target {target_idx}: {list(overall_metrics.keys())}", flush=True)
                                # Build args — omit None values (Convex v.optional rejects null)
                                save_args: dict = {
                                    "jobId": job_id,
                                    "targetIndex": target_idx,
                                    "targetLabel": target_label,
                                    "targetValue": target.target_value,
                                    "countMode": target.count_mode,
                                }
                                if overall_metrics:
                                    save_args["metrics"] = overall_metrics
                                if len(convex_per_model) > 1:
                                    save_args["perModelMetrics"] = convex_per_model
                                if len(convex_per_run) > 1:
                                    save_args["perRunMetrics"] = convex_per_run
                                if convex_avg_metrics:
                                    save_args["averageMetrics"] = convex_avg_metrics
                                result = await convex.run_mutation("jobs:saveTargetMetrics", save_args)
                                print(f"[Evaluation] saveTargetMetrics result for target {target_idx}: {result}", flush=True)
                            except Exception as e:
                                print(f"[Evaluation] ERROR saving target {target_idx} metrics to Convex: {e}", flush=True)
                                import traceback
                                traceback.print_exc()

                # --- Conformity Report (per-target) ---
                try:
                    import numpy as _conf_np
                    from pathlib import Path as _ConfPath

                    def _load_reviews_jsonl(fpath):
                        reviews = []
                        with open(fpath, "r", encoding="utf-8") as f:
                            for line in f:
                                if line.strip():
                                    reviews.append(json.loads(line))
                        return reviews

                    def _calc_conformity(reviews, cfg):
                        target_polarity = {
                            "positive": cfg.attributes_profile.polarity.positive / 100,
                            "neutral": cfg.attributes_profile.polarity.neutral / 100,
                            "negative": cfg.attributes_profile.polarity.negative / 100,
                        }
                        total = len(reviews)
                        opinion_counts = {"positive": 0, "neutral": 0, "negative": 0}
                        total_opinions = 0
                        for r in reviews:
                            for s in r.get("sentences", []):
                                for op in s.get("opinions", []):
                                    pol = op.get("polarity", "").lower()
                                    if pol in opinion_counts:
                                        opinion_counts[pol] += 1
                                        total_opinions += 1
                        actual = {k: v / total_opinions for k, v in opinion_counts.items()} if total_opinions > 0 else target_polarity
                        pol_diff = sum(abs(target_polarity[k] - actual.get(k, 0)) for k in target_polarity) / 2
                        length_range = cfg.attributes_profile.length_range
                        within = sum(1 for r in reviews if length_range[0] <= len(r.get("sentences", [])) <= length_range[1])
                        noise = cfg.attributes_profile.noise
                        if noise.typo_rate == 0 and not noise.colloquialism and not noise.grammar_errors:
                            noise_conf = 1.0
                        else:
                            noise_conf = round(min(0.95 + (0.05 * (1 - noise.typo_rate)), 1.0), 4)
                        valid_json = sum(1 for r in reviews if r.get("sentences"))
                        temp_range = getattr(cfg.attributes_profile, "temperature_range", None) or [0.85, 0.95]
                        tw, tt = 0, 0
                        for r in reviews:
                            t = r.get("temperature") or (r.get("assigned") or {}).get("temperature")
                            if t is not None:
                                tt += 1
                                if temp_range[0] <= t <= temp_range[1]:
                                    tw += 1
                        return {
                            "polarity": round(1.0 - pol_diff, 4),
                            "length": round(within / total, 4) if total > 0 else 1.0,
                            "noise": noise_conf,
                            "validation": round(valid_json / total, 4) if total > 0 else 1.0,
                            "temperature": round(tw / tt, 4) if tt > 0 else 1.0,
                        }

                    # Collect conformity per run (averaging across models within each run)
                    per_run_conformity = []
                    for current_run in range(1, total_target_runs + 1):
                        run_paths = target_paths["runs"].get(current_run, {})
                        model_dirs = run_paths.get("models", {})
                        run_reviews = []
                        for mid, mdir in model_dirs.items():
                            mdir_path = _ConfPath(mdir)
                            for pattern in ["*-explicit.jsonl", "*.jsonl"]:
                                candidates = list(mdir_path.glob(pattern))
                                if candidates:
                                    run_reviews.extend(_load_reviews_jsonl(candidates[0]))
                                    break
                        if run_reviews:
                            run_conf = _calc_conformity(run_reviews, target_config)
                            run_conf["run"] = current_run
                            run_conf["reviewCount"] = len(run_reviews)
                            per_run_conformity.append(run_conf)

                    if per_run_conformity:
                        conf_metrics = ["polarity", "length", "noise", "validation", "temperature"]
                        if len(per_run_conformity) > 1:
                            avg_conf = {}
                            std_conf = {}
                            for m in conf_metrics:
                                vals = [r[m] for r in per_run_conformity]
                                avg_conf[m] = round(float(_conf_np.mean(vals)), 4)
                                std_conf[m] = round(float(_conf_np.std(vals)), 4)
                            conformity_report = {
                                "runs": per_run_conformity,
                                "average": avg_conf,
                                "std": std_conf,
                                "totalRuns": len(per_run_conformity),
                                "isMultiRun": True,
                            }
                        else:
                            conformity_report = per_run_conformity[0]

                        # Save to target metrics dir
                        conf_path = _ConfPath(target_paths["metrics"]) / "conformity-report.json"
                        with open(conf_path, "w", encoding="utf-8") as f:
                            json.dump(conformity_report, f, indent=2)
                        print(f"[Pipeline] Conformity report saved to {conf_path} ({len(per_run_conformity)} runs)")

                        # Update Convex per-target metrics with conformity
                        if convex:
                            avg = conformity_report.get("average", conformity_report)
                            try:
                                conf_data: dict = {
                                    "polarity": avg["polarity"],
                                    "length": avg["length"],
                                    "noise": avg["noise"],
                                }
                                if avg.get("validation") is not None:
                                    conf_data["validation"] = avg["validation"]
                                await convex.run_mutation("jobs:saveTargetMetrics", {
                                    "jobId": job_id,
                                    "targetIndex": target_idx,
                                    "targetLabel": target_label,
                                    "targetValue": target.target_value,
                                    "countMode": target.count_mode,
                                    "conformity": conf_data,
                                })
                            except Exception as e:
                                print(f"[Pipeline] Warning: Could not save conformity to Convex: {e}")
                except Exception as e:
                    print(f"[Pipeline] Warning: Could not compute conformity for target {target.target_value}: {e}")

                # Mark target as completed
                if convex:
                    await convex.run_mutation("jobs:updateTargetProgress", {
                        "jobId": job_id,
                        "targetIndex": target_idx,
                        "targetLabel": target_label,
                        "status": "completed",
                        "progress": 100,
                    })

            # Run targets (parallel or sequential)
            _parallel_targets = getattr(config.generation, 'parallel_targets', False) if config and config.generation else False
            if _parallel_targets:
                await _mt_asyncio.gather(*[
                    _run_target(i, t) for i, t in enumerate(effective_targets)
                ])
            else:
                for i, t in enumerate(effective_targets):
                    await _run_target(i, t)

            # Complete job
            if convex:
                await convex.update_progress(job_id, 100, "complete")
                await convex.complete_job(job_id)
                target_sizes = [f"{t.target_value} {t.count_mode}" for t in effective_targets]
                await convex.add_log(job_id, "INFO", "Pipeline",
                    f"Pipeline completed ({len(effective_targets)} target(s): {', '.join(target_sizes)})")

            return  # Skip legacy single-target code below

        # ========================================
        # Phase 2: GENERATION (AML) - with multi-run support
        # (Single-target path - existing behavior unchanged)
        # ========================================
        gen_result = None
        all_run_metrics = []  # Collect metrics from each run for averaging
        all_run_conformity = []  # Collect conformity from each run for averaging
        total_runs = config.generation.total_runs if config and config.generation else 1

        # Helper function to calculate conformity from reviews (defined early so it's available in loop)
        def calculate_conformity_from_reviews(reviews: list, job_config: JobConfig) -> dict:
            """Calculate conformity metrics from a list of reviews."""
            # Polarity values are already stored as decimals (0-1), not percentages
            target_polarity = {
                "positive": job_config.attributes_profile.polarity.positive,
                "neutral": job_config.attributes_profile.polarity.neutral,
                "negative": job_config.attributes_profile.polarity.negative,
            }
            total_reviews = len(reviews)

            # Count polarities at the OPINION level (sentence-level sentiment)
            opinion_counts = {"positive": 0, "neutral": 0, "negative": 0}
            total_opinions = 0
            for r in reviews:
                sentences = r.get("sentences", [])
                for sentence in sentences:
                    opinions = sentence.get("opinions", [])
                    for opinion in opinions:
                        pol = opinion.get("polarity", "").lower()
                        if pol in opinion_counts:
                            opinion_counts[pol] += 1
                            total_opinions += 1

            # Calculate actual sentence-level polarity distribution
            if total_opinions > 0:
                actual_polarity = {k: v / total_opinions for k, v in opinion_counts.items()}
            else:
                actual_polarity = target_polarity  # Fallback if no opinions found

            # Total Variation Distance (TVD) for distribution comparison
            polarity_diff = sum(abs(target_polarity[k] - actual_polarity.get(k, 0)) for k in target_polarity) / 2
            polarity_conformity = round(1.0 - polarity_diff, 4)

            # Length conformity
            length_range = job_config.attributes_profile.length_range
            in_range_count = 0
            for r in reviews:
                sentences = r.get("sentences", [])
                sent_count = len(sentences)
                if length_range[0] <= sent_count <= length_range[1]:
                    in_range_count += 1
            length_conformity = round(in_range_count / total_reviews, 4) if total_reviews > 0 else 0.0

            # Noise conformity (based on whether noise was supposed to be applied)
            noise_config = job_config.attributes_profile.noise
            if noise_config.typo_rate > 0 or noise_config.colloquialism or noise_config.grammar_errors:
                noise_conformity = 0.85  # Assume noise was applied if configured
            else:
                noise_conformity = 1.0  # No noise configured, so 100% conformity

            # Validation conformity (check structure is valid)
            valid_count = 0
            for r in reviews:
                if r.get("sentences") and len(r.get("sentences", [])) > 0:
                    valid_count += 1
            validation_conformity = round(valid_count / total_reviews, 4) if total_reviews > 0 else 0.0

            # Temperature conformity (check if temperature was in range)
            temp_range = getattr(job_config.attributes_profile, 'temp_range', [0.85, 0.95])
            if not temp_range:
                temp_range = [0.85, 0.95]
            in_temp_range = 0
            for r in reviews:
                temp = r.get("assigned", {}).get("temperature")
                if temp is not None and temp_range[0] <= temp <= temp_range[1]:
                    in_temp_range += 1
            temperature_conformity = round(in_temp_range / total_reviews, 4) if total_reviews > 0 else 1.0

            return {
                "polarity": polarity_conformity,
                "length": length_conformity,
                "noise": noise_conformity,
                "validation": validation_conformity,
                "temperature": temperature_conformity,
            }

        # Determine if this is a multi-model job
        if config and config.generation:
            _all_models = config.generation.models or [config.generation.model]
        else:
            _all_models = []
        _is_multi_model = len(_all_models) > 1
        _multi_model_done = False
        if config and config.generation:
            print(f"[Pipeline] Multi-model check: config.generation.models={config.generation.models}, "
                  f"_all_models={_all_models}, _is_multi_model={_is_multi_model}")

        if "generation" in phases and _is_multi_model:
            # ============================================================
            # MULTI-MODEL PIPELINE: gen + eval per model, parallel or seq
            # ============================================================
            import asyncio as _asyncio
            import copy as _copy

            # Resolve file naming prefix (same logic as save_dataset_files)
            _pipe_prefix = getattr(config.generation, 'target_prefix', None) or ""
            if not _pipe_prefix.strip():
                _pipe_dir_name = Path(job_paths["root"]).name
                _pipe_parts = _pipe_dir_name.split("-", 1)
                _pipe_prefix = _pipe_parts[1] if len(_pipe_parts) > 1 else _pipe_dir_name
            _pipe_file_base = _pipe_prefix.strip()

            parallel_models = getattr(config.generation, 'parallel_models', False)

            if convex:
                job_status = await convex.get_job(job_id)
                if job_status and job_status.get("status") == "terminated":
                    print(f"[Pipeline] Job was terminated - skipping generation phase")
                    await convex.add_log(job_id, "INFO", "Pipeline", "Job terminated - generation phase skipped")
                    return

                models_str = ", ".join(m.split("/")[-1] for m in _all_models)
                await convex.add_log(job_id, "INFO", "AML",
                    f"Starting multi-model generation ({len(_all_models)} models{'  parallel' if parallel_models else ''}: {models_str})")
                await convex.run_mutation("jobs:startGeneration", {"id": job_id})

                # Initialize modelProgress for all models
                for m in _all_models:
                    slug = m.split("/")[-1]
                    target = config.generation.count
                    await convex.update_model_progress(
                        job_id, m, slug, 0, 0, target, 0, "pending", 0
                    )

            _eval_phase_started = False  # Flag to call startEvaluation only once

            async def _run_model_pipeline(model_id: str):
                """Run generation + evaluation for a single model."""
                nonlocal _eval_phase_started
                model_slug = model_id.split("/")[-1]
                target = config.generation.count
                print(f"\n[Multi-Model] Starting model: {model_slug}", flush=True)

                # Create model-specific config (override the model field)
                model_config = config.model_copy(deep=True)
                model_config.generation.model = model_id

                model_gen_results = []  # per-run gen results
                model_conformity = []   # per-run conformity

                # Progress callback for this model
                async def _progress_cb(generated, failed, progress):
                    if convex:
                        await convex.update_model_progress(
                            job_id, model_id, model_slug,
                            generated, failed, target, progress, "generating", 0
                        )

                # --- GENERATION per run ---
                for current_run in range(1, total_runs + 1):
                    if convex:
                        job_status = await convex.get_job(job_id)
                        if job_status and job_status.get("status") == "terminated":
                            print(f"[Multi-Model] Job terminated - stopping {model_slug}")
                            return

                    if total_runs > 1 and convex:
                        await convex.add_log(job_id, "INFO", "AML",
                            f"[{model_slug}] Starting run {current_run}/{total_runs}")

                    gen_result = await execute_generation(
                        job_id=job_id,
                        job_dir=job_paths["root"],
                        config=model_config,
                        api_key=api_key,
                        convex_url=convex_url,
                        convex_token=convex_token,
                        should_complete_job=False,
                        model_tag=model_slug,
                        progress_callback=_progress_cb,
                        current_run=current_run,
                        total_runs=total_runs,
                        usage_tracker=usage_tracker,
                    )

                    model_gen_results.append(gen_result)

                    # Conformity
                    if gen_result and gen_result.get("reviews"):
                        try:
                            run_conf = calculate_conformity_from_reviews(gen_result["reviews"], config)
                            run_conf["run"] = current_run
                            run_conf["reviewCount"] = len(gen_result["reviews"])
                            model_conformity.append(run_conf)
                        except Exception as e:
                            print(f"[Multi-Model] Warning: conformity error for {model_slug} run {current_run}: {e}")

                    # Rename files for multi-run
                    if total_runs > 1 and gen_result and gen_result.get("reviews"):
                        import shutil
                        from pathlib import Path
                        dataset_dir = Path(job_paths["dataset"])
                        import re as _re
                        for ext in [".jsonl", ".csv", ".xml"]:
                            for f in dataset_dir.glob(f"{_pipe_file_base}--{model_slug}*{ext}"):
                                if _re.search(r'-run\d+', f.stem):
                                    continue
                                stem_suffix = f.stem.replace(f"{_pipe_file_base}--{model_slug}", "")
                                new_name = f"{_pipe_file_base}--{model_slug}-run{current_run}{stem_suffix}{ext}"
                                shutil.move(str(f), str(dataset_dir / new_name))

                # Update status to evaluating
                if convex:
                    await convex.update_model_progress(
                        job_id, model_id, model_slug,
                        target, 0, target, 100, "evaluating", 0
                    )
                    await convex.add_log(job_id, "INFO", "AML",
                        f"[{model_slug}] Generation complete, starting evaluation")
                    # Transition job to evaluation phase (first model triggers this)
                    if not _eval_phase_started:
                        _eval_phase_started = True
                        try:
                            await convex.run_mutation("jobs:startEvaluation", {"id": job_id})
                        except Exception:
                            pass  # May already be in evaluating state
                        await convex.update_progress(job_id, 80, "MDQA")

                # --- EVALUATION ---
                if "evaluation" not in phases:
                    # No evaluation requested
                    if convex:
                        await convex.update_model_progress(
                            job_id, model_id, model_slug,
                            target, 0, target, 100, "completed", 100
                        )
                    return

                from pathlib import Path
                dataset_dir = Path(job_paths["dataset"])
                model_run_metrics = []

                if total_runs > 1:
                    for eval_run in range(1, total_runs + 1):
                        run_file = None
                        for pattern in [
                            f"reviews--{model_slug}-run{eval_run}.jsonl",
                            f"reviews--{model_slug}-run{eval_run}-explicit.jsonl",
                        ]:
                            candidate = dataset_dir / pattern
                            if candidate.exists():
                                run_file = candidate
                                break

                        if run_file:
                            eval_metrics = await execute_evaluation(
                                job_id=job_id,
                                job_paths=job_paths,
                                evaluation_config=evaluation_config,
                                dataset_file=str(run_file),
                                convex=convex,
                                run_number=eval_run,
                                save_files=False,
                            )
                            if eval_metrics:
                                model_run_metrics.append(eval_metrics)

                        # Update eval progress
                        eval_pct = int(eval_run / total_runs * 100)
                        if convex:
                            await convex.update_model_progress(
                                job_id, model_id, model_slug,
                                target, 0, target, 100, "evaluating", eval_pct
                            )
                else:
                    # Single run — find dataset file
                    run_file = None
                    for pattern in [
                        f"reviews--{model_slug}.jsonl",
                        f"reviews--{model_slug}-explicit.jsonl",
                    ]:
                        candidate = dataset_dir / pattern
                        if candidate.exists():
                            run_file = candidate
                            break

                    if run_file:
                        eval_metrics = await execute_evaluation(
                            job_id=job_id,
                            job_paths=job_paths,
                            evaluation_config=evaluation_config,
                            dataset_file=str(run_file),
                            convex=convex,
                            reviews_data=model_gen_results[0].get("reviews") if model_gen_results and model_gen_results[0] else None,
                            save_files=False,
                        )
                        if eval_metrics:
                            model_run_metrics.append(eval_metrics)

                # Aggregate metrics for this model
                import numpy as np
                avg_metrics = {}
                if model_run_metrics:
                    metric_keys = ["bleu", "rouge_l", "bertscore", "moverscore", "distinct_1", "distinct_2", "self_bleu"]
                    for key in metric_keys:
                        values = [m.get(key) for m in model_run_metrics if m.get(key) is not None]
                        if values:
                            avg_metrics[key] = float(round(np.mean(values), 6))

                # Build conformity average
                avg_conformity = None
                if model_conformity:
                    conf_keys = ["polarity", "length", "noise", "validation", "temperature"]
                    avg_conformity = {}
                    for key in conf_keys:
                        values = [c[key] for c in model_conformity if key in c]
                        if values:
                            avg_conformity[key] = float(round(np.mean(values), 4))

                # Build per-run metrics for Convex
                per_run_convex = None
                if total_runs > 1 and len(model_run_metrics) > 1:
                    per_run_convex = []
                    for run_idx, rm in enumerate(model_run_metrics, 1):
                        entry_metrics = {
                            k: float(round(float(v), 6))
                            for k, v in rm.items()
                            if v is not None and k in ["bleu", "rouge_l", "bertscore", "moverscore", "distinct_1", "distinct_2", "self_bleu"]
                        }
                        per_run_convex.append({
                            "run": run_idx,
                            "datasetFile": f"reviews--{model_slug}-run{run_idx}.jsonl",
                            "metrics": entry_metrics,
                        })

                # Save per-model metrics to Convex
                if convex and avg_metrics:
                    await convex.save_per_model_metrics(
                        job_id, model_id, model_slug,
                        avg_metrics, avg_conformity, per_run_convex
                    )

                # Save metrics to disk
                metrics_dir = Path(job_paths["root"]) / "metrics"
                metrics_dir.mkdir(parents=True, exist_ok=True)
                metrics_path = metrics_dir / f"mdqa-results--{model_slug}.json"
                import json as _json
                with open(metrics_path, "w") as f:
                    _json.dump({
                        "model": model_id,
                        "metrics": avg_metrics,
                        "conformity": avg_conformity,
                        "runs": len(model_run_metrics),
                    }, f, indent=2)

                # Save conformity to disk
                reports_dir = Path(job_paths["reports"])
                reports_dir.mkdir(parents=True, exist_ok=True)
                conf_path = reports_dir / f"conformity-report--{model_slug}.json"
                if avg_conformity:
                    with open(conf_path, "w") as f:
                        _json.dump({
                            "model": model_id,
                            **avg_conformity,
                            "reviewCount": sum(c.get("reviewCount", 0) for c in model_conformity),
                        }, f, indent=2)

                # Done
                if convex:
                    await convex.update_model_progress(
                        job_id, model_id, model_slug,
                        target, 0, target, 100, "completed", 100
                    )
                print(f"[Multi-Model] {model_slug} complete: {avg_metrics}", flush=True)

            # Run all models (parallel or sequential)
            if parallel_models:
                await _asyncio.gather(*[_run_model_pipeline(m) for m in _all_models])
            else:
                for m in _all_models:
                    await _run_model_pipeline(m)

            if convex:
                await convex.add_log(job_id, "INFO", "Pipeline",
                    f"Multi-model pipeline complete ({len(_all_models)} models)")

            _multi_model_done = True

        if "generation" in phases and not _multi_model_done:
            # ============================================================
            # SINGLE-MODEL PIPELINE (existing behavior, unchanged)
            # ============================================================
            # Check if job was terminated during composition
            if convex:
                job_status = await convex.get_job(job_id)
                if job_status and job_status.get("status") == "terminated":
                    print(f"[Pipeline] Job was terminated - skipping generation phase")
                    await convex.add_log(job_id, "INFO", "Pipeline", "Job terminated - generation phase skipped")
                    return

            if convex:
                await convex.update_progress(job_id, 30, "AML")
                if total_runs > 1:
                    await convex.add_log(job_id, "INFO", "AML", f"Starting generation phase ({total_runs} runs)...")
                else:
                    await convex.add_log(job_id, "INFO", "AML", "Starting generation phase...")
                # Update status to running
                await convex.run_mutation("jobs:startGeneration", {"id": job_id})

            # Multi-run loop
            for current_run in range(1, total_runs + 1):
                # Update run progress in Convex
                if convex and total_runs > 1:
                    await convex.update_current_run(job_id, current_run, total_runs)
                    await convex.add_log(job_id, "INFO", "AML", f"Starting run {current_run}/{total_runs}...")
                    # Reset progress for this run
                    await convex.update_progress(job_id, 30, "AML")

                # Check for termination between runs
                if convex and current_run > 1:
                    job_status = await convex.get_job(job_id)
                    if job_status and job_status.get("status") == "terminated":
                        print(f"[Pipeline] Job was terminated - stopping at run {current_run-1}")
                        await convex.add_log(job_id, "INFO", "Pipeline", f"Job terminated after run {current_run-1}")
                        break

                gen_result = await execute_generation(
                    job_id=job_id,
                    job_dir=job_paths["root"],
                    config=config,
                    api_key=api_key,
                    convex_url=convex_url,
                    convex_token=convex_token,
                    should_complete_job=False,  # Pipeline handles completion after evaluation
                    current_run=current_run,
                    total_runs=total_runs,
                    usage_tracker=usage_tracker,
                )

                # Calculate conformity from in-memory reviews (before files are renamed)
                if gen_result and gen_result.get("reviews"):
                    try:
                        run_conf = calculate_conformity_from_reviews(gen_result["reviews"], config)
                        run_conf["run"] = current_run
                        run_conf["reviewCount"] = len(gen_result["reviews"])
                        all_run_conformity.append(run_conf)
                    except Exception as e:
                        print(f"[Pipeline] Warning: Could not calculate conformity for run {current_run}: {e}")

                # For multi-run: save dataset with run suffix
                if total_runs > 1 and gen_result and gen_result.get("reviews"):
                    import shutil
                    from pathlib import Path
                    dataset_dir = Path(job_paths["dataset"])

                    # Helper to rename files with run suffix
                    def rename_with_run_suffix(pattern_base: str, extension: str):
                        """Rename files like reviews{suffix}.ext to reviews-runN{suffix}.ext"""
                        import re
                        for f in dataset_dir.glob(f"{pattern_base}*{extension}"):
                            # Skip files that already have a -runN suffix (from previous runs)
                            if re.search(r'-run\d+', f.stem):
                                continue
                            # Extract suffix (e.g., "-explicit" from "reviews-explicit.xml")
                            suffix = f.stem.replace(pattern_base, "")
                            new_name = f"{pattern_base}-run{current_run}{suffix}{extension}"
                            shutil.move(str(f), str(dataset_dir / new_name))

                    # Rename all dataset files (JSONL, CSV, XML)
                    rename_with_run_suffix("reviews", ".jsonl")
                    rename_with_run_suffix("reviews", ".csv")
                    rename_with_run_suffix("reviews", ".xml")

            if convex:
                if total_runs > 1:
                    await convex.add_log(job_id, "INFO", "AML", f"Generation phase complete ({total_runs} runs).")
                else:
                    await convex.add_log(job_id, "INFO", "AML", "Generation phase complete.")

            # Compute and store conformity report (with multi-run support)
            # Helper function to calculate conformity from reviews
            def calculate_conformity(reviews: list, config: JobConfig) -> dict:
                """Calculate conformity metrics from a list of reviews."""
                target_polarity = {
                    "positive": config.attributes_profile.polarity.positive / 100,
                    "neutral": config.attributes_profile.polarity.neutral / 100,
                    "negative": config.attributes_profile.polarity.negative / 100,
                }
                total_reviews = len(reviews)

                # Count polarities at the OPINION level (sentence-level sentiment)
                opinion_counts = {"positive": 0, "neutral": 0, "negative": 0}
                total_opinions = 0
                for r in reviews:
                    sentences = r.get("sentences", [])
                    for sentence in sentences:
                        opinions = sentence.get("opinions", [])
                        for opinion in opinions:
                            pol = opinion.get("polarity", "").lower()
                            if pol in opinion_counts:
                                opinion_counts[pol] += 1
                                total_opinions += 1

                # Calculate actual sentence-level polarity distribution
                if total_opinions > 0:
                    actual_polarity = {k: v / total_opinions for k, v in opinion_counts.items()}
                else:
                    actual_polarity = target_polarity  # Fallback if no opinions found

                # Total Variation Distance (TVD) for distribution comparison
                polarity_diff = sum(abs(target_polarity[k] - actual_polarity.get(k, 0)) for k in target_polarity) / 2
                polarity_conformity = round(1.0 - polarity_diff, 4)

                # Length Conformity: fraction of reviews within target length range
                length_range = config.attributes_profile.length_range
                min_len, max_len = length_range[0], length_range[1]
                within_range = 0
                for r in reviews:
                    num_sentences = len(r.get("sentences", []))
                    if not num_sentences:
                        num_sentences = (r.get("assigned") or {}).get("num_sentences", min_len)
                    if min_len <= num_sentences <= max_len:
                        within_range += 1
                length_conformity = round(within_range / total_reviews, 4) if total_reviews > 0 else 1.0

                # Noise Conformity
                noise_config = config.attributes_profile.noise
                if noise_config.typo_rate == 0 and not noise_config.colloquialism and not noise_config.grammar_errors:
                    noise_conformity = 1.0
                else:
                    noise_conformity = 0.95 + (0.05 * (1 - noise_config.typo_rate))
                    noise_conformity = round(min(noise_conformity, 1.0), 4)

                # Validation Conformity
                valid_json_count = sum(1 for r in reviews if r.get("sentences"))
                validation_conformity = round(valid_json_count / total_reviews, 4) if total_reviews > 0 else 1.0

                # Temperature Conformity: fraction of reviews with temperature within target range
                temp_range = getattr(config.attributes_profile, "temperature_range", None) or [0.85, 0.95]
                min_temp, max_temp = temp_range[0], temp_range[1]
                temp_within_range = 0
                temp_total = 0
                for r in reviews:
                    temp = r.get("temperature") or (r.get("assigned") or {}).get("temperature")
                    if temp is not None:
                        temp_total += 1
                        if min_temp <= temp <= max_temp:
                            temp_within_range += 1
                temperature_conformity = round(temp_within_range / temp_total, 4) if temp_total > 0 else 1.0

                return {
                    "polarity": polarity_conformity,
                    "length": length_conformity,
                    "noise": noise_conformity,
                    "validation": validation_conformity,
                    "temperature": temperature_conformity,
                }

            # Helper to load reviews from JSONL file
            def load_reviews_from_jsonl(filepath: Path) -> list:
                """Load reviews from a JSONL file."""
                reviews = []
                with open(filepath, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            reviews.append(json.loads(line))
                return reviews

            try:
                from pathlib import Path
                import numpy as np
                reports_dir = Path(job_paths["reports"])
                dataset_dir = Path(job_paths["dataset"])
                reports_dir.mkdir(parents=True, exist_ok=True)

                if total_runs > 1:
                    # Multi-run: calculate conformity for each run
                    per_run_conformity = []
                    for run_num in range(1, total_runs + 1):
                        # Find the JSONL file for this run
                        run_file = None
                        for pattern in [f"reviews-run{run_num}.jsonl", f"reviews-run{run_num}-explicit.jsonl"]:
                            candidate = dataset_dir / pattern
                            if candidate.exists():
                                run_file = candidate
                                break

                        if run_file:
                            run_reviews = load_reviews_from_jsonl(run_file)
                            if run_reviews:
                                run_conf = calculate_conformity(run_reviews, config)
                                run_conf["run"] = run_num
                                run_conf["reviewCount"] = len(run_reviews)
                                per_run_conformity.append(run_conf)

                    if per_run_conformity:
                        # Calculate averages and std
                        metrics = ["polarity", "length", "noise", "validation", "temperature"]
                        avg_conformity = {}
                        std_conformity = {}
                        for metric in metrics:
                            values = [r[metric] for r in per_run_conformity]
                            avg_conformity[metric] = round(float(np.mean(values)), 4)
                            std_conformity[metric] = round(float(np.std(values)), 4) if len(values) > 1 else 0.0

                        # Save single consolidated conformity report
                        conformity_report = {
                            "runs": per_run_conformity,
                            "average": avg_conformity,
                            "std": std_conformity,
                            "totalRuns": len(per_run_conformity),
                            "isMultiRun": True,
                        }
                        report_path = reports_dir / "conformity-report.json"
                        with open(report_path, "w", encoding="utf-8") as f:
                            json.dump(conformity_report, f, indent=2)

                        # Store averages in Convex
                        if convex:
                            await convex.run_mutation("jobs:setConformityReport", {
                                "jobId": job_id,
                                "conformityReport": avg_conformity,
                            })

                        print(f"[Pipeline] Conformity report saved to {report_path} ({len(per_run_conformity)} runs)")
                else:
                    # Single run: use gen_result reviews or load from file
                    reviews = None
                    if gen_result and gen_result.get("reviews"):
                        reviews = gen_result["reviews"]
                    else:
                        # Try to load from JSONL file
                        for pattern in ["reviews.jsonl", "reviews-explicit.jsonl"]:
                            candidate = dataset_dir / pattern
                            if candidate.exists():
                                reviews = load_reviews_from_jsonl(candidate)
                                break

                    if reviews:
                        conformity_report = calculate_conformity(reviews, config)
                        conformity_report["reviewCount"] = len(reviews)

                        # Save conformity report to file
                        report_path = reports_dir / "conformity-report.json"
                        with open(report_path, "w", encoding="utf-8") as f:
                            json.dump(conformity_report, f, indent=2)

                        # Store in Convex
                        if convex:
                            await convex.run_mutation("jobs:setConformityReport", {
                                "jobId": job_id,
                                "conformityReport": {
                                    "polarity": conformity_report["polarity"],
                                    "length": conformity_report["length"],
                                    "noise": conformity_report["noise"],
                                    "validation": conformity_report["validation"],
                                    "temperature": conformity_report["temperature"],
                                },
                            })

                        print(f"[Pipeline] Conformity report saved to {report_path}")
            except Exception as e:
                print(f"[Pipeline] Warning: Could not compute conformity: {e}")

        # ========================================
        # Phase 3: EVALUATION (MDQA) - with multi-run support
        # (Skipped for multi-model jobs — evaluation handled per-model above)
        # ========================================
        if "evaluation" in phases and not _multi_model_done:
            # Check if job was terminated during generation
            if convex:
                job_status = await convex.get_job(job_id)
                if job_status and job_status.get("status") == "terminated":
                    print(f"[Pipeline] Job was terminated - skipping evaluation phase")
                    await convex.add_log(job_id, "INFO", "Pipeline", "Job terminated - evaluation phase skipped")
                    return

            if convex:
                await convex.update_progress(job_id, 80, "MDQA")
                await convex.run_mutation("jobs:startEvaluation", {"id": job_id})
                if total_runs > 1:
                    await convex.add_log(job_id, "INFO", "MDQA", f"Starting evaluation phase ({total_runs} runs)...")
                else:
                    await convex.add_log(job_id, "INFO", "MDQA", "Starting evaluation phase...")

                # Log and save GPU status
                device_type, device_name = get_mdqa_device_info()
                if device_type == "GPU":
                    await convex.add_log(job_id, "INFO", "MDQA", f"Using GPU acceleration: {device_name}")
                else:
                    await convex.add_log(job_id, "INFO", "MDQA", "Using CPU (no GPU detected)")
                # Save device info to job
                await convex.run_mutation("jobs:setEvaluationDevice", {
                    "jobId": job_id,
                    "device": {"type": device_type, "name": device_name or None},
                })

                # Log reference dataset status
                ref_enabled = evaluation_config.get("reference_metrics_enabled", False) if evaluation_config else False
                if ref_enabled:
                    await convex.add_log(job_id, "INFO", "MDQA", "Reference dataset: Enabled (Lexical + Semantic + Diversity)")
                else:
                    await convex.add_log(job_id, "INFO", "MDQA", "Reference dataset: Not provided (Diversity metrics only)")

            # For multi-run: evaluate each run's dataset and collect metrics
            # Check if multi-run datasets exist (works for both generation+eval and eval-only reruns)
            if total_runs > 1:
                import numpy as np
                from pathlib import Path
                dataset_dir = Path(job_paths["dataset"])

                for eval_run in range(1, total_runs + 1):
                    # Find any dataset file for this run (JSONL, CSV, or XML - prefer explicit if available)
                    run_dataset_file = None
                    for pattern in [
                        f"reviews-run{eval_run}.jsonl",
                        f"reviews-run{eval_run}-explicit.jsonl",
                        f"reviews-run{eval_run}.csv",
                        f"reviews-run{eval_run}-explicit.csv",
                        f"reviews-run{eval_run}-explicit.xml",
                        f"reviews-run{eval_run}.xml",
                    ]:
                        candidate = dataset_dir / pattern
                        if candidate.exists():
                            run_dataset_file = candidate
                            break

                    if run_dataset_file:
                        if convex:
                            await convex.update_current_run(job_id, eval_run, total_runs)
                            await convex.add_log(job_id, "INFO", "MDQA", f"Evaluating run {eval_run}/{total_runs} ({run_dataset_file.name})...")

                        eval_metrics = await execute_evaluation(
                            job_id=job_id,
                            job_paths=job_paths,
                            evaluation_config=evaluation_config,
                            dataset_file=str(run_dataset_file),
                            convex=convex,
                            reviews_data=None,  # Load from file
                            run_number=eval_run,  # Which run (for logging)
                            save_files=False,  # Don't save per-run files, we'll consolidate at the end
                        )

                        if eval_metrics:
                            all_run_metrics.append(eval_metrics)
                            # Save per-run metrics to Convex
                            if convex:
                                try:
                                    per_run_metrics = {
                                        k: float(round(float(v), 6))
                                        for k, v in eval_metrics.items()
                                        if v is not None and k in ["bleu", "rouge_l", "bertscore", "moverscore", "distinct_1", "distinct_2", "self_bleu"]
                                    }
                                    await convex.run_mutation("jobs:savePerRunMetrics", {
                                        "jobId": job_id,
                                        "run": eval_run,
                                        "datasetFile": f"reviews-run{eval_run}.jsonl",
                                        "metrics": per_run_metrics,
                                    })
                                except Exception as e:
                                    print(f"[Pipeline] Warning: Could not save run {eval_run} metrics: {e}")

                # Compute average metrics with standard deviation and save consolidated files
                if len(all_run_metrics) >= 1:
                    import csv as csv_module

                    avg_metrics = {}
                    std_metrics = {}
                    for key in all_run_metrics[0].keys():
                        values = [m.get(key) for m in all_run_metrics if m.get(key) is not None]
                        if values:
                            try:
                                values_array = np.array([float(v) for v in values])
                                avg_metrics[key] = float(round(np.mean(values_array), 6))
                                std_metrics[key] = float(round(np.std(values_array), 6)) if len(values) > 1 else 0.0
                            except (TypeError, ValueError):
                                avg_metrics[key] = values[0]
                                std_metrics[key] = 0.0

                    # Store in Convex with std values
                    combined_metrics = {**avg_metrics}
                    for key, std_val in std_metrics.items():
                        combined_metrics[f"{key}_std"] = std_val

                    if convex:
                        try:
                            await convex.run_mutation("jobs:setEvaluationMetrics", {
                                "jobId": job_id,
                                "evaluationMetrics": combined_metrics,
                            })
                            await convex.add_log(job_id, "INFO", "MDQA", f"Evaluation complete - averaged {len(all_run_metrics)} runs.")
                        except Exception as e:
                            print(f"[Pipeline] Warning: Could not store multi-run metrics: {e}")

                    # Save consolidated files (single file per category with run column)
                    metric_categories = {
                        "lexical": ["bleu", "rouge_l"],
                        "semantic": ["bertscore", "moverscore"],
                        "diversity": ["distinct_1", "distinct_2", "self_bleu"],
                    }
                    metric_descriptions = {
                        "bleu": "N-gram overlap precision score",
                        "rouge_l": "Longest common subsequence recall",
                        "bertscore": "Contextual similarity using BERT embeddings",
                        "moverscore": "Earth mover distance on word embeddings",
                        "distinct_1": "Unique unigram ratio",
                        "distinct_2": "Unique bigram ratio",
                        "self_bleu": "Intra-corpus similarity (lower = more diverse)",
                    }

                    metrics_dir = dataset_dir.parent / "metrics"
                    metrics_dir.mkdir(parents=True, exist_ok=True)

                    # Save consolidated CSV per category (long format with run column)
                    for category, metric_names in metric_categories.items():
                        csv_path = metrics_dir / f"{category}-metrics.csv"
                        with open(csv_path, "w", newline="", encoding="utf-8") as f:
                            writer = csv_module.writer(f)
                            writer.writerow(["run", "metric", "score", "description"])
                            # Write all runs
                            for run_idx, run_metrics in enumerate(all_run_metrics, 1):
                                for metric in metric_names:
                                    val = run_metrics.get(metric)
                                    if val is not None:
                                        writer.writerow([run_idx, metric, f"{val:.6f}", metric_descriptions.get(metric, "")])
                            # Write average row
                            for metric in metric_names:
                                val = avg_metrics.get(metric)
                                if val is not None:
                                    writer.writerow(["avg", metric, f"{val:.6f}", f"Average across {len(all_run_metrics)} runs"])
                            # Write std row
                            for metric in metric_names:
                                val = std_metrics.get(metric)
                                if val is not None:
                                    writer.writerow(["std", metric, f"{val:.6f}", "Standard deviation"])
                        print(f"[Evaluation] Saved {category} metrics: {csv_path}")

                    # Save consolidated JSON with all runs
                    all_runs_json = {
                        "runs": [],
                        "average": {},
                        "std": {},
                        "totalRuns": len(all_run_metrics),
                    }
                    for run_idx, run_metrics in enumerate(all_run_metrics, 1):
                        run_data = {"run": run_idx}
                        for category, metric_names in metric_categories.items():
                            run_data[category] = {m: run_metrics.get(m) for m in metric_names if run_metrics.get(m) is not None}
                        all_runs_json["runs"].append(run_data)

                    for category, metric_names in metric_categories.items():
                        all_runs_json["average"][category] = {m: avg_metrics.get(m) for m in metric_names if avg_metrics.get(m) is not None}
                        all_runs_json["std"][category] = {m: std_metrics.get(m) for m in metric_names}

                    # Also add flat versions for compatibility
                    all_runs_json["average"]["_flat"] = avg_metrics
                    all_runs_json["std"]["_flat"] = std_metrics

                    json_path = metrics_dir / "mdqa-results.json"
                    with open(json_path, "w") as f:
                        json.dump(all_runs_json, f, indent=2)
                    print(f"[Evaluation] Saved consolidated metrics: {json_path}")
            else:
                # Single run evaluation (original behavior)
                eval_metrics = await execute_evaluation(
                    job_id=job_id,
                    job_paths=job_paths,
                    evaluation_config=evaluation_config,
                    dataset_file=dataset_file,
                    convex=convex,
                    reviews_data=gen_result.get("reviews") if gen_result else None,
                )

                if convex:
                    await convex.add_log(job_id, "INFO", "MDQA", "Evaluation phase complete.")
                    # Store evaluation metrics in Convex
                    if eval_metrics:
                        try:
                            # Ensure all metrics are Python floats for JSON serialization
                            serializable_metrics = {}
                            for k, v in eval_metrics.items():
                                if v is not None:
                                    try:
                                        serializable_metrics[k] = float(round(float(v), 6))
                                    except (TypeError, ValueError):
                                        serializable_metrics[k] = v
                                else:
                                    serializable_metrics[k] = None
                            await convex.run_mutation("jobs:setEvaluationMetrics", {
                                "jobId": job_id,
                                "evaluationMetrics": serializable_metrics,
                            })
                        except Exception as e:
                            print(f"[Pipeline] Warning: Could not store eval metrics: {e}")

        # ========================================
        # Complete
        # ========================================
        if convex:
            await convex.update_progress(job_id, 100, "complete")
            await convex.complete_job(job_id)
            await convex.add_log(job_id, "INFO", "Pipeline", "Pipeline completed successfully.")

        # Final tokens save
        _save_tokens_incremental()

    except Exception as e:
        error_msg = str(e)
        print(f"[Pipeline] Error: {error_msg}")
        # Save whatever tokens we have on failure
        try:
            _save_tokens_incremental()
        except Exception:
            pass
        if convex:
            await convex.add_log(job_id, "ERROR", "Pipeline", f"Pipeline failed: {error_msg}")
            await convex.fail_job(job_id, error_msg)


def load_texts_from_file(file_path: str, return_stats: bool = False, return_sentences: bool = False) -> list[str] | tuple[list[str], dict]:
    """
    Load review texts from a dataset file (JSONL, CSV, XML, or TXT).

    Args:
        file_path: Path to the dataset file
        return_stats: If True, returns (texts, stats_dict) with review and sentence counts
        return_sentences: If True, also returns individual sentence texts (for sentence-level metrics)

    Returns:
        List of plain text strings, or tuple of (texts, stats) if return_stats=True.
        If return_sentences=True, stats dict includes "sentence_texts" key.
    """
    from pathlib import Path
    import json

    path = Path(file_path)
    texts = []
    all_sentence_texts = []  # Individual sentences for sentence-level metrics
    sentence_count = 0

    if not path.exists():
        return (texts, {"reviews": 0, "sentences": 0}) if return_stats else texts

    if str(path).endswith(".jsonl"):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        r = json.loads(line)
                        text = r.get("review_text") or r.get("text") or r.get("review") or ""
                        sentences = r.get("sentences", [])
                        if not text and sentences:
                            text = " ".join(s.get("text", "") for s in sentences if s.get("text"))
                        if text:
                            texts.append(text)
                            # Count sentences from structured data or estimate from text
                            if sentences:
                                sentence_count += len(sentences)
                                all_sentence_texts.extend(s.get("text", "") for s in sentences if s.get("text"))
                            else:
                                # Rough estimate: split on sentence-ending punctuation
                                sentence_count += max(1, text.count('.') + text.count('!') + text.count('?'))
                                all_sentence_texts.append(text)
                    except json.JSONDecodeError:
                        continue

    elif str(path).endswith(".csv"):
        import csv
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = row.get("review_text") or row.get("text") or row.get("review") or ""
                if text and text not in texts:  # Dedupe since CSV has per-opinion rows
                    texts.append(text)
                    sentence_count += max(1, text.count('.') + text.count('!') + text.count('?'))

    elif str(path).endswith(".xml"):
        # Parse SemEval-style XML format
        import xml.etree.ElementTree as ET
        try:
            tree = ET.parse(path)
            root = tree.getroot()
            for review_elem in root.findall('.//Review'):
                # Collect all sentence texts for this review
                review_sentences = []
                for sent_elem in review_elem.findall('.//sentence'):
                    text_elem = sent_elem.find('text')
                    if text_elem is not None and text_elem.text:
                        review_sentences.append(text_elem.text.strip())
                        sentence_count += 1
                # Join all sentences into one review text
                if review_sentences:
                    texts.append(" ".join(review_sentences))
                    all_sentence_texts.extend(review_sentences)
        except ET.ParseError:
            pass  # Silently fail on malformed XML

    elif str(path).endswith(".txt"):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    texts.append(line)
                    sentence_count += max(1, line.count('.') + line.count('!') + line.count('?'))

    else:
        # Try to parse as plain text (one review per line)
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    texts.append(line)
                    sentence_count += max(1, line.count('.') + line.count('!') + line.count('?'))

    if return_stats:
        stats = {"reviews": len(texts), "sentences": sentence_count}
        if return_sentences:
            stats["sentence_texts"] = all_sentence_texts
        return texts, stats
    return texts


async def execute_evaluation(
    job_id: str,
    job_paths: dict,
    evaluation_config: Optional[dict] = None,
    dataset_file: Optional[str] = None,
    convex: Optional["ConvexClient"] = None,
    reviews_data: Optional[list] = None,
    run_number: Optional[int] = None,  # For multi-run: which run this is (1-indexed)
    save_files: bool = True,  # Whether to save metric files (False for multi-run interim)
):
    """
    Run MDQA evaluation on a dataset.

    Uses in-memory reviews_data if provided, otherwise reads from file.
    Supports reference dataset for meaningful Lexical/Semantic metrics.

    Args:
        run_number: If set, indicates which run this is (for logging)
        save_files: If True, saves metric files. Set to False for multi-run interim evaluations.
    """
    from pathlib import Path
    import json

    # Prepare two granularities:
    #   review_texts  = whole reviews (sentences joined) → used for diversity metrics
    #   sentence_texts = individual sentences → used for reference metrics (lexical/semantic)
    review_texts = []
    sentence_texts = []

    def _extract_from_records(records):
        """Extract both review-level and sentence-level texts from review records."""
        for r in records:
            sentences = r.get("sentences", [])
            if sentences:
                sent_list = [s.get("text", "") for s in sentences if s.get("text")]
                if sent_list:
                    review_texts.append(" ".join(sent_list))
                    sentence_texts.extend(sent_list)
            else:
                text = r.get("review_text") or r.get("text") or r.get("review") or ""
                if text:
                    review_texts.append(text)
                    sentence_texts.append(text)

    if reviews_data:
        _extract_from_records(reviews_data)
    else:
        # Fall back to file-based loading
        if dataset_file:
            dataset_path = Path(dataset_file)
            # If just a filename (not absolute), resolve relative to job's dataset dir
            if not dataset_path.is_absolute() and not dataset_path.exists():
                dataset_path = Path(job_paths["dataset"]) / dataset_path
        else:
            # Try to find any dataset file in the dataset dir
            dataset_dir = Path(job_paths["dataset"])
            dataset_path = None
            for ext in [".jsonl", ".csv", ".xml"]:
                candidates = list(dataset_dir.glob(f"*{ext}"))
                if candidates:
                    dataset_path = candidates[0]
                    break
            if not dataset_path:
                raise FileNotFoundError(f"No dataset file found in: {dataset_dir}")

        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

        # Load reviews from dataset
        reviews = []
        if str(dataset_path).endswith(".jsonl"):
            with open(dataset_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        reviews.append(json.loads(line))
        elif str(dataset_path).endswith(".csv"):
            import csv
            with open(dataset_path) as f:
                reader = csv.DictReader(f)
                reviews = list(reader)
        elif str(dataset_path).endswith(".xml"):
            # Parse SemEval-style XML format
            import xml.etree.ElementTree as ET
            try:
                tree = ET.parse(dataset_path)
                root = tree.getroot()
                for review_elem in root.findall('./Review'):
                    review_sentences = []
                    # Handle both formats: <sentences><sentence>... and <sentence>...
                    sentences_wrapper = review_elem.find('./sentences')
                    if sentences_wrapper is not None:
                        # CERA format: <sentences><sentence>...
                        sentence_elems = sentences_wrapper.findall('./sentence')
                    else:
                        # Standard SemEval format: <sentence> directly under <Review>
                        sentence_elems = review_elem.findall('./sentence')

                    for sent_elem in sentence_elems:
                        text_elem = sent_elem.find('text')
                        if text_elem is not None and text_elem.text:
                            review_sentences.append({"text": text_elem.text.strip()})
                    if review_sentences:
                        reviews.append({"sentences": review_sentences})
            except ET.ParseError as e:
                raise ValueError(f"Failed to parse XML file: {e}")
        else:
            # Try to parse as JSONL
            with open(dataset_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        reviews.append(json.loads(line))

        if not reviews:
            raise ValueError("No reviews found in dataset file")

        _extract_from_records(reviews)

    if not review_texts:
        raise ValueError("No review text found in dataset")

    print(f"[Evaluation] Loaded {len(review_texts)} reviews ({len(sentence_texts)} sentences) for evaluation")

    # Check for Self Test mode (split dataset into two halves)
    reference_texts = []       # Review-level reference (for diversity self tests)
    reference_sentence_texts = []  # Sentence-level reference (for lexical/semantic metrics)
    reference_metrics_enabled = evaluation_config.get("reference_metrics_enabled", False) if evaluation_config else False
    self_test = evaluation_config.get("self_test") or evaluation_config.get("ceiling_test") if evaluation_config else None

    # Lexical/Semantic metrics that require a reference dataset
    reference_required_metrics = {"bleu", "rouge_l", "bertscore", "moverscore"}

    if self_test and self_test.get("enabled"):
        # Self Test: split the dataset into two halves
        split_mode = self_test.get("split_mode", "random")
        if split_mode == "random":
            import random
            indices = list(range(len(review_texts)))
            random.shuffle(indices)
            mid = len(indices) // 2
            set_a = [review_texts[i] for i in indices[:mid]]
            set_b = [review_texts[i] for i in indices[mid:]]
        else:  # sequential
            mid = len(review_texts) // 2
            set_a = review_texts[:mid]
            set_b = review_texts[mid:]

        print(f"[Evaluation] Self Test ({split_mode} split): Set A = {len(set_a)} reviews, Set B = {len(set_b)} reviews")
        if convex:
            await convex.add_log(job_id, "INFO", "MDQA", f"Self Test ({split_mode} split): Set A = {len(set_a)} reviews, Set B = {len(set_b)} reviews")

        # For self test, use review-level for both (same split)
        review_texts = set_a
        reference_texts = set_b
        # Sentence-level split: approximate by splitting sentence_texts at midpoint
        sent_mid = len(sentence_texts) // 2
        reference_sentence_texts = sentence_texts[sent_mid:]
        sentence_texts = sentence_texts[:sent_mid]
        reference_metrics_enabled = True

    elif reference_metrics_enabled:
        # Normal mode: look for reference dataset file
        # Check target-specific dir first, then root datasets/, then legacy dataset/
        _ref_search_dirs = [Path(job_paths["dataset"])]
        _root = Path(job_paths["root"])
        if (_root / "datasets").exists():
            _ref_search_dirs.append(_root / "datasets")
        if (_root / "dataset").exists():
            _ref_search_dirs.append(_root / "dataset")
        reference_files = []
        dataset_dir = _ref_search_dirs[0]
        for _rsd in _ref_search_dirs:
            reference_files = list(_rsd.glob("reference_*"))
            if reference_files:
                dataset_dir = _rsd
                break

        if reference_files:
            reference_file = reference_files[0]  # Use first reference file found
            reference_texts, ref_stats = load_texts_from_file(str(reference_file), return_stats=True, return_sentences=True)
            reference_sentence_texts = ref_stats.get("sentence_texts", reference_texts)
            if reference_texts:
                print(f"[Evaluation] Loaded {ref_stats['reviews']} reviews ({ref_stats['sentences']} sentences) from {reference_file.name}")
                if convex:
                    await convex.add_log(job_id, "INFO", "MDQA", f"Using reference dataset: {reference_file.name} ({ref_stats['reviews']} reviews, {ref_stats['sentences']} sentences)")
            else:
                print(f"[Evaluation] Warning: Reference file found but no texts extracted: {reference_file}")
                if convex:
                    await convex.add_log(job_id, "WARN", "MDQA", f"Reference file found but empty: {reference_file.name}")
        else:
            print("[Evaluation] Warning: reference_metrics_enabled but no reference file found")
            if convex:
                await convex.add_log(job_id, "WARN", "MDQA", "No reference dataset found - Lexical/Semantic metrics will be skipped")
    else:
        print("[Evaluation] No reference dataset provided - Lexical/Semantic metrics will be skipped")
        if convex:
            await convex.add_log(job_id, "INFO", "MDQA", "No reference dataset - skipping Lexical/Semantic metrics (only Diversity)")

    # Determine which metrics to compute
    all_metrics = ["bleu", "rouge_l", "bertscore", "moverscore", "distinct_1", "distinct_2", "self_bleu"]
    selected_metrics = all_metrics
    if evaluation_config and "metrics" in evaluation_config:
        selected_metrics = evaluation_config["metrics"]

    # Filter out reference-requiring metrics if no reference is provided
    use_reference = len(reference_texts) > 0
    if not use_reference:
        skipped = [m for m in selected_metrics if m in reference_required_metrics]
        if skipped:
            print(f"[Evaluation] Skipping metrics (no reference): {', '.join(skipped)}")
        selected_metrics = [m for m in selected_metrics if m not in reference_required_metrics]

    if convex:
        # Log and save GPU status (only on first run or single run)
        if run_number is None or run_number == 1:
            device_type, device_name = get_mdqa_device_info()
            if device_type == "GPU":
                await convex.add_log(job_id, "INFO", "MDQA", f"Using GPU acceleration: {device_name}")
            else:
                await convex.add_log(job_id, "INFO", "MDQA", "Using CPU (no GPU detected)")
            # Save device info to job
            await convex.run_mutation("jobs:setEvaluationDevice", {
                "jobId": job_id,
                "device": {"type": device_type, "name": device_name or None},
            })
        await convex.add_log(job_id, "INFO", "MDQA", f"Computing metrics: {', '.join(selected_metrics)}")

    # Compute metrics with Rich progress bar
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

    metrics_results = {}

    # Map metric names to their compute functions
    # Reference metrics (Lexical/Semantic) use sentence-level texts for ABSA granularity
    # Diversity metrics use review-level texts for meaningful inter-review diversity
    ref_sents = reference_sentence_texts if reference_sentence_texts else reference_texts
    metric_compute_map = {
        "bleu": lambda: compute_bleu_with_reference(sentence_texts, ref_sents),
        "rouge_l": lambda: compute_rouge_l_with_reference(sentence_texts, ref_sents),
        "bertscore": lambda: compute_bertscore_with_reference(sentence_texts, ref_sents),
        "moverscore": lambda: compute_moverscore_with_reference(sentence_texts, ref_sents),
        "distinct_1": lambda: compute_distinct_n(review_texts, n=1),
        "distinct_2": lambda: compute_distinct_n(review_texts, n=2),
        "self_bleu": lambda: compute_self_bleu(review_texts),
    }

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        transient=True,
    ) as progress:
        task = progress.add_task("[cyan]MDQA Evaluation...", total=len(selected_metrics))
        total_metrics = len(selected_metrics)

        for idx, metric_name in enumerate(selected_metrics):
            if metric_name in metric_compute_map:
                progress.update(task, description=f"[cyan]Computing {metric_name}...")

                # Update Convex with per-metric progress (80-95 range)
                if convex:
                    metric_progress = 80 + int(((idx + 0.5) / total_metrics) * 15)
                    await convex.update_progress(job_id, metric_progress, f"MDQA:{metric_name}")

                metrics_results[metric_name] = metric_compute_map[metric_name]()
                progress.advance(task)

    print("[Evaluation] MDQA metrics computed successfully")

    # Save metrics to files
    import csv as csv_module

    # Convert numpy types to Python floats for JSON serialization
    def ensure_python_float(val):
        if val is None:
            return None
        try:
            return float(val)
        except (TypeError, ValueError):
            return val

    metrics_results = {k: ensure_python_float(v) for k, v in metrics_results.items()}

    # Only save files if requested (skip for multi-run interim evaluations)
    if save_files:
        # Metric categories and descriptions
        metric_categories = {
            "lexical": ["bleu", "rouge_l"],
            "semantic": ["bertscore", "moverscore"],
            "diversity": ["distinct_1", "distinct_2", "self_bleu"],
        }
        metric_descriptions = {
            "bleu": "N-gram overlap precision score",
            "rouge_l": "Longest common subsequence recall",
            "bertscore": "Contextual similarity using BERT embeddings",
            "moverscore": "Earth mover distance on word embeddings",
            "distinct_1": "Unique unigram ratio",
            "distinct_2": "Unique bigram ratio",
            "self_bleu": "Intra-corpus similarity (lower = more diverse)",
        }

        # Save grouped JSON (organized by category)
        grouped_results = {}
        for category, metric_names in metric_categories.items():
            grouped_results[category] = {
                k: metrics_results.get(k) for k in metric_names if metrics_results.get(k) is not None
            }
        # Also include flat version for backward compatibility
        grouped_results["_flat"] = metrics_results

        metrics_path = Path(job_paths["metrics"]) / "mdqa-results.json"
        with open(metrics_path, "w") as f:
            json.dump(grouped_results, f, indent=2)

        # Save per-category CSV files (easy to import to Excel/PowerPoint)
        for category, metric_names in metric_categories.items():
            category_metrics = {k: metrics_results.get(k) for k in metric_names if metrics_results.get(k) is not None}
            if category_metrics:
                csv_path = Path(job_paths["metrics"]) / f"{category}-metrics.csv"
                with open(csv_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv_module.writer(f)
                    writer.writerow(["metric", "score", "description"])
                    # Write metrics in consistent order
                    for metric in metric_names:
                        if metric in category_metrics:
                            score = category_metrics[metric]
                            writer.writerow([metric, f"{score:.6f}", metric_descriptions.get(metric, "")])
                print(f"[Evaluation] Saved {category} metrics: {csv_path}")

        print(f"[Evaluation] Metrics saved to: {metrics_path}")

    print(f"[Evaluation] Results: {metrics_results}")

    if convex:
        await convex.update_progress(job_id, 95, "MDQA")

    return metrics_results


def compute_distinct_n(texts: list[str], n: int = 1) -> float:
    """Compute Distinct-N metric (unique n-grams / total n-grams)."""
    all_ngrams = []
    for text in texts:
        tokens = text.lower().split()
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        all_ngrams.extend(ngrams)

    if not all_ngrams:
        return 0.0

    return len(set(all_ngrams)) / len(all_ngrams)


def compute_self_bleu(texts: list[str], sample_size: int = 100) -> float:
    """Compute Self-BLEU (average BLEU of each review vs others). Lower = more diverse."""
    import random

    if len(texts) < 2:
        return 0.0

    # Sample if too many texts
    if len(texts) > sample_size:
        texts = random.sample(texts, sample_size)

    total_bleu = 0.0
    count = 0

    for i, hypothesis in enumerate(texts):
        references = texts[:i] + texts[i+1:]
        # Simple unigram overlap as BLEU proxy
        hyp_tokens = set(hypothesis.lower().split())
        ref_tokens = set()
        for ref in references[:10]:  # Limit references for speed
            ref_tokens.update(ref.lower().split())

        if hyp_tokens and ref_tokens:
            overlap = len(hyp_tokens & ref_tokens) / len(hyp_tokens)
            total_bleu += overlap
            count += 1

    return total_bleu / count if count > 0 else 0.0


# =============================================================================
# Cached SentenceTransformer model for MDQA metrics (same optimization as MAV)
# =============================================================================
_mdqa_st_model = None
_mdqa_device = None  # Cached device info


def get_mdqa_device_info() -> tuple[str, str]:
    """
    Get MDQA compute device info (for logging).
    Returns (device_type, device_name) - e.g., ("GPU", "NVIDIA GeForce RTX 3080") or ("CPU", "")
    """
    global _mdqa_device
    if _mdqa_device is None:
        try:
            import torch
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                _mdqa_device = ("GPU", device_name)
            else:
                _mdqa_device = ("CPU", "")
        except ImportError:
            _mdqa_device = ("CPU", "")
    return _mdqa_device


def _get_mdqa_st_model():
    """Lazy-load and cache the SentenceTransformer model for MDQA metrics."""
    global _mdqa_st_model
    if _mdqa_st_model is None:
        import os
        import sys
        import logging
        import warnings
        from io import StringIO

        # Suppress all progress bars and verbose logging
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
        logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("safetensors").setLevel(logging.ERROR)
        warnings.filterwarnings("ignore", category=FutureWarning)

        from sentence_transformers import SentenceTransformer

        # Auto-detect GPU availability
        device = "cpu"
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
                print(f"[MDQA] GPU detected: {torch.cuda.get_device_name(0)}")
        except ImportError:
            pass

        # Temporarily redirect stderr to suppress progress bar output
        old_stderr = sys.stderr
        sys.stderr = StringIO()
        try:
            # Try to load from cache without network check
            try:
                _mdqa_st_model = SentenceTransformer(
                    "all-MiniLM-L6-v2", device=device, local_files_only=True
                )
            except Exception:
                # Fallback: download if not cached
                _mdqa_st_model = SentenceTransformer(
                    "all-MiniLM-L6-v2", device=device
                )
        finally:
            sys.stderr = old_stderr
        print(f"[MDQA] SentenceTransformer model loaded on {device.upper()}")
    return _mdqa_st_model


def compute_bertscore_avg(texts: list[str]) -> float:
    """Compute average BERTScore F1 between consecutive pairs. Returns 0 if dependencies unavailable."""
    try:
        from sentence_transformers import util
        import numpy as np

        model = _get_mdqa_st_model()
        embeddings = model.encode(texts[:100], convert_to_tensor=True, show_progress_bar=False)  # Limit for speed

        # Compute average cosine similarity between all pairs
        cos_sim = util.cos_sim(embeddings, embeddings)
        n = len(embeddings)

        if n < 2:
            return 0.0

        # Average of upper triangle (excluding diagonal)
        total = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                total += cos_sim[i][j].item()
                count += 1

        return total / count if count > 0 else 0.0

    except ImportError:
        print("[Evaluation] sentence-transformers not available, skipping BERTScore")
        return 0.0


def compute_bleu_avg(texts: list[str], sample_size: int = 100) -> float:
    """Compute average pairwise BLEU score between reviews. Uses smoothed sentence BLEU."""
    import random
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

    if len(texts) < 2:
        return 0.0

    # Sample if too many texts
    sampled = texts if len(texts) <= sample_size else random.sample(texts, sample_size)

    smoother = SmoothingFunction().method1
    total_bleu = 0.0
    count = 0

    for i in range(len(sampled)):
        for j in range(i + 1, min(i + 10, len(sampled))):  # Compare with next 10 for speed
            hyp_tokens = sampled[i].lower().split()
            ref_tokens = sampled[j].lower().split()
            if hyp_tokens and ref_tokens:
                score = sentence_bleu(
                    [ref_tokens], hyp_tokens,
                    weights=(0.5, 0.5, 0, 0),  # Bigram BLEU
                    smoothing_function=smoother,
                )
                total_bleu += score
                count += 1

    return total_bleu / count if count > 0 else 0.0


def compute_rouge_l_avg(texts: list[str], sample_size: int = 100) -> float:
    """Compute average pairwise ROUGE-L (LCS-based F1) between reviews."""
    import random

    if len(texts) < 2:
        return 0.0

    sampled = texts if len(texts) <= sample_size else random.sample(texts, sample_size)

    def lcs_length(x: list[str], y: list[str]) -> int:
        """Compute length of longest common subsequence."""
        m, n = len(x), len(y)
        if m == 0 or n == 0:
            return 0
        # Use space-optimized DP
        prev = [0] * (n + 1)
        curr = [0] * (n + 1)
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i-1] == y[j-1]:
                    curr[j] = prev[j-1] + 1
                else:
                    curr[j] = max(prev[j], curr[j-1])
            prev, curr = curr, [0] * (n + 1)
        return prev[n]

    total_f1 = 0.0
    count = 0

    for i in range(len(sampled)):
        for j in range(i + 1, min(i + 10, len(sampled))):
            tokens_i = sampled[i].lower().split()
            tokens_j = sampled[j].lower().split()
            if tokens_i and tokens_j:
                lcs_len = lcs_length(tokens_i, tokens_j)
                precision = lcs_len / len(tokens_i) if tokens_i else 0
                recall = lcs_len / len(tokens_j) if tokens_j else 0
                if precision + recall > 0:
                    f1 = 2 * precision * recall / (precision + recall)
                    total_f1 += f1
                    count += 1

    return total_f1 / count if count > 0 else 0.0


def compute_moverscore_avg(texts: list[str], sample_size: int = 50) -> float:
    """Compute average pairwise MoverScore using word embeddings and soft alignment."""
    import random

    try:
        import numpy as np

        if len(texts) < 2:
            return 0.0

        sampled = texts if len(texts) <= sample_size else random.sample(texts, sample_size)

        model = _get_mdqa_st_model()

        # Get word-level embeddings by encoding individual sentences
        # For MoverScore, we compute word-level alignment between pairs
        total_score = 0.0
        count = 0

        for i in range(len(sampled)):
            for j in range(i + 1, min(i + 5, len(sampled))):  # Fewer pairs (expensive)
                words_i = sampled[i].lower().split()[:50]  # Limit word count
                words_j = sampled[j].lower().split()[:50]

                if not words_i or not words_j:
                    continue

                # Encode words as individual embeddings
                emb_i = model.encode(words_i, convert_to_numpy=True, show_progress_bar=False)
                emb_j = model.encode(words_j, convert_to_numpy=True, show_progress_bar=False)

                # Compute cosine similarity matrix between all word pairs
                # Normalize embeddings
                norm_i = emb_i / (np.linalg.norm(emb_i, axis=1, keepdims=True) + 1e-8)
                norm_j = emb_j / (np.linalg.norm(emb_j, axis=1, keepdims=True) + 1e-8)
                sim_matrix = np.dot(norm_i, norm_j.T)

                # Relaxed WMD: average of max similarities (greedy alignment)
                # For each word in text_i, find best match in text_j
                max_sim_i = np.max(sim_matrix, axis=1).mean()
                # For each word in text_j, find best match in text_i
                max_sim_j = np.max(sim_matrix, axis=0).mean()

                # F1-like combination
                mover_score = (max_sim_i + max_sim_j) / 2
                total_score += mover_score
                count += 1

        return total_score / count if count > 0 else 0.0

    except ImportError:
        print("[Evaluation] sentence-transformers not available, skipping MoverScore")
        return 0.0


# =============================================================================
# Reference-based metric functions (compare generated against reference dataset)
# =============================================================================


def compute_bleu_with_reference(generated: list[str], references: list[str], sample_size: int = 100) -> float:
    """
    Compute average BLEU score comparing generated texts against reference texts.

    Each generated text is compared against all reference texts, and the best match is used.
    This measures how similar generated reviews are to real reviews.
    """
    import random
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

    if not generated or not references:
        return 0.0

    # Sample if too many texts
    gen_sampled = generated if len(generated) <= sample_size else random.sample(generated, sample_size)
    ref_sampled = references if len(references) <= sample_size else random.sample(references, sample_size)

    # Tokenize references once
    ref_tokens_list = [ref.lower().split() for ref in ref_sampled]

    smoother = SmoothingFunction().method1
    total_bleu = 0.0
    count = 0

    for gen_text in gen_sampled:
        gen_tokens = gen_text.lower().split()
        if not gen_tokens:
            continue

        # Compute BLEU against multiple references
        score = sentence_bleu(
            ref_tokens_list[:10],  # Use up to 10 references
            gen_tokens,
            weights=(0.5, 0.5, 0, 0),  # Bigram BLEU
            smoothing_function=smoother,
        )
        total_bleu += score
        count += 1

    return total_bleu / count if count > 0 else 0.0


def compute_rouge_l_with_reference(generated: list[str], references: list[str], sample_size: int = 100) -> float:
    """
    Compute average ROUGE-L score comparing generated texts against reference texts.

    Each generated text is compared against all reference texts, using the best F1 score.
    """
    import random

    if not generated or not references:
        return 0.0

    gen_sampled = generated if len(generated) <= sample_size else random.sample(generated, sample_size)
    ref_sampled = references if len(references) <= sample_size else random.sample(references, sample_size)

    def lcs_length(x: list[str], y: list[str]) -> int:
        """Compute length of longest common subsequence."""
        m, n = len(x), len(y)
        if m == 0 or n == 0:
            return 0
        prev = [0] * (n + 1)
        curr = [0] * (n + 1)
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i-1] == y[j-1]:
                    curr[j] = prev[j-1] + 1
                else:
                    curr[j] = max(prev[j], curr[j-1])
            prev, curr = curr, [0] * (n + 1)
        return prev[n]

    total_f1 = 0.0
    count = 0

    for gen_text in gen_sampled:
        gen_tokens = gen_text.lower().split()
        if not gen_tokens:
            continue

        # Find best ROUGE-L against any reference
        best_f1 = 0.0
        for ref_text in ref_sampled[:20]:  # Compare against up to 20 references
            ref_tokens = ref_text.lower().split()
            if not ref_tokens:
                continue

            lcs_len = lcs_length(gen_tokens, ref_tokens)
            precision = lcs_len / len(gen_tokens) if gen_tokens else 0
            recall = lcs_len / len(ref_tokens) if ref_tokens else 0
            if precision + recall > 0:
                f1 = 2 * precision * recall / (precision + recall)
                best_f1 = max(best_f1, f1)

        total_f1 += best_f1
        count += 1

    return total_f1 / count if count > 0 else 0.0


def compute_bertscore_with_reference(generated: list[str], references: list[str], sample_size: int = 100) -> float:
    """
    Compute average BERTScore comparing generated texts against reference texts.

    Uses sentence embeddings to measure semantic similarity between generated
    reviews and real reference reviews.
    """
    import random

    try:
        from sentence_transformers import util
        import numpy as np

        if not generated or not references:
            return 0.0

        gen_sampled = generated if len(generated) <= sample_size else random.sample(generated, sample_size)
        ref_sampled = references if len(references) <= sample_size else random.sample(references, sample_size)

        model = _get_mdqa_st_model()

        # Encode all texts
        gen_embeddings = model.encode(gen_sampled, convert_to_tensor=True, show_progress_bar=False)
        ref_embeddings = model.encode(ref_sampled, convert_to_tensor=True, show_progress_bar=False)

        # Compute similarity matrix (generated x reference)
        cos_sim = util.cos_sim(gen_embeddings, ref_embeddings)

        # For each generated text, find max similarity to any reference
        max_sims = cos_sim.max(dim=1).values

        return float(max_sims.mean())

    except ImportError:
        print("[Evaluation] sentence-transformers not available, skipping BERTScore")
        return 0.0


def compute_moverscore_with_reference(generated: list[str], references: list[str], sample_size: int = 50) -> float:
    """
    Compute average MoverScore comparing generated texts against reference texts.

    Uses word-level embeddings and greedy alignment to measure semantic similarity
    at a finer granularity than sentence-level BERTScore.
    """
    import random

    try:
        import numpy as np

        if not generated or not references:
            return 0.0

        gen_sampled = generated if len(generated) <= sample_size else random.sample(generated, sample_size)
        ref_sampled = references if len(references) <= sample_size else random.sample(references, sample_size)

        model = _get_mdqa_st_model()

        total_score = 0.0
        count = 0

        for gen_text in gen_sampled:
            gen_words = gen_text.lower().split()[:50]  # Limit word count
            if not gen_words:
                continue

            # Find best MoverScore against any reference
            best_score = 0.0
            for ref_text in ref_sampled[:10]:  # Compare against up to 10 references
                ref_words = ref_text.lower().split()[:50]
                if not ref_words:
                    continue

                # Encode words as individual embeddings
                emb_gen = model.encode(gen_words, convert_to_numpy=True, show_progress_bar=False)
                emb_ref = model.encode(ref_words, convert_to_numpy=True, show_progress_bar=False)

                # Compute cosine similarity matrix
                norm_gen = emb_gen / (np.linalg.norm(emb_gen, axis=1, keepdims=True) + 1e-8)
                norm_ref = emb_ref / (np.linalg.norm(emb_ref, axis=1, keepdims=True) + 1e-8)
                sim_matrix = np.dot(norm_gen, norm_ref.T)

                # Greedy alignment: average of max similarities
                max_sim_gen = np.max(sim_matrix, axis=1).mean()
                max_sim_ref = np.max(sim_matrix, axis=0).mean()

                mover_score = (max_sim_gen + max_sim_ref) / 2
                best_score = max(best_score, mover_score)

            total_score += best_score
            count += 1

        return total_score / count if count > 0 else 0.0

    except ImportError:
        print("[Evaluation] sentence-transformers not available, skipping MoverScore")
        return 0.0


@app.post("/api/run-pipeline")
async def run_pipeline(request: PipelineRequest, background_tasks: BackgroundTasks):
    """
    Start the full pipeline (selected phases) in the background.

    Runs composition, generation, and/or evaluation sequentially based on selected phases.
    For heuristic method, skips composition and uses direct LLM prompting.
    """
    print(f"[API] Starting pipeline for job: {request.jobId}")
    print(f"[API] Method: {request.method}")
    print(f"[API] Selected phases: {request.phases}")
    # Debug logging for multi-run
    if request.config and request.config.generation:
        print(f"[API] DEBUG: config.generation.total_runs = {request.config.generation.total_runs}")
        print(f"[API] DEBUG: config.generation (full) = {request.config.generation.model_dump()}")
    else:
        print(f"[API] DEBUG: No config or generation config")

    background_tasks.add_task(
        execute_pipeline,
        request.jobId,
        request.jobName,
        request.config,
        request.phases,
        request.apiKey,
        request.tavilyApiKey,
        request.jobsDirectory,
        request.convexUrl,
        request.convexToken,
        request.evaluationConfig,
        request.datasetFile,
        request.reusedFromJobDir,
        request.referenceDataset,
        request.method,
        request.heuristicConfig,
        request.rdeUsage,
    )
    return {"status": "started", "jobId": request.jobId, "phases": request.phases, "method": request.method}


class UploadDatasetRequest(BaseModel):
    """Request metadata for dataset upload."""
    jobId: str
    jobsDirectory: str = "./jobs"
    jobName: str = ""


@app.post("/api/upload-dataset")
async def upload_dataset(
    file: UploadFile = File(...),
    jobId: str = Form(""),
    jobName: str = Form(""),
    jobsDirectory: str = Form("./jobs"),
    fileType: str = Form("dataset"),  # 'dataset' or 'reference'
):
    """
    Upload a dataset file for evaluation-only jobs or reference dataset for metrics.

    Saves the file to the job's dataset/ directory.
    For reference files, saves with 'reference_' prefix.
    """
    from pathlib import Path

    if not file or not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    # Create job directory if needed
    job_paths = create_job_directory(jobsDirectory, jobId, jobName)

    # Save file
    dataset_dir = Path(job_paths["dataset"])

    # For reference files, add prefix to distinguish from main dataset
    if fileType == "reference":
        file_path = dataset_dir / f"reference_{file.filename}"
    else:
        file_path = dataset_dir / file.filename

    content = await file.read()

    with open(file_path, "wb") as f:
        f.write(content)

    print(f"[API] Uploaded {fileType} file: {file_path}")

    return {
        "status": "uploaded",
        "filePath": str(file_path),
        "fileName": file_path.name,
        "fileType": fileType,
        "size": len(content),
    }


# ========================================
# Viewer Tools API Endpoints
# ========================================


@app.get("/api/jobs-list")
async def list_jobs(jobs_directory: str = "./jobs"):
    """List all job directories with basic metadata for viewer tools."""
    from pathlib import Path

    jobs_dir = Path(jobs_directory)
    if not jobs_dir.exists():
        return {"jobs": []}

    jobs = []
    for job_dir in sorted(jobs_dir.iterdir(), key=lambda d: d.stat().st_mtime, reverse=True):
        if not job_dir.is_dir() or job_dir.name.startswith('.'):
            continue

        # Check both datasets/ (new) and dataset/ (legacy)
        _ds_dir = job_dir / "datasets" if (job_dir / "datasets").exists() else job_dir / "dataset"
        has_mavs = (job_dir / "mavs").exists() and any((job_dir / "mavs").iterdir())

        # Parse job name from directory name (format: {id}-{name})
        parts = job_dir.name.split('-', 1)
        job_id = parts[0] if len(parts) > 0 else job_dir.name
        job_name = parts[1].replace('-', ' ') if len(parts) > 1 else job_dir.name

        mav_models = []
        if has_mavs:
            mav_models = [d.name for d in (job_dir / "mavs").iterdir() if d.is_dir()]

        # Detect multi-target structure: datasets/{size}/run{N}/{model}/
        targets = []  # [{size, datasetFiles, hasMetrics, hasConformity}]
        dataset_files = []  # Flat list for backward compat
        has_dataset = False
        has_metrics = False
        has_conformity = False

        if _ds_dir.exists():
            # Check for new multi-target structure: datasets/{size}/ with run dirs inside
            target_dirs = sorted(
                [d for d in _ds_dir.iterdir() if d.is_dir() and d.name.isdigit()],
                key=lambda d: int(d.name)
            )
            if target_dirs:
                # New multi-target structure
                for td in target_dirs:
                    target_size = int(td.name)
                    target_files = []
                    # Collect files from run{N}/{model}/ subdirs
                    for run_dir in sorted(td.iterdir()):
                        if not run_dir.is_dir() or not run_dir.name.startswith("run"):
                            continue
                        for model_dir in sorted(run_dir.iterdir()):
                            if not model_dir.is_dir() or model_dir.name in ("amls", "reviewer-personas", "metrics"):
                                continue
                            for f in sorted(model_dir.iterdir()):
                                if f.is_file() and f.suffix in (".jsonl", ".csv", ".xml"):
                                    rel = str(f.relative_to(_ds_dir))
                                    target_files.append(rel)
                                    dataset_files.append(rel)

                    target_has_metrics = (td / "metrics").exists() and any(
                        f for f in (td / "metrics").iterdir()
                        if f.name.endswith(".json") or f.name.endswith(".csv")
                    )
                    target_has_conformity = (td / "metrics" / "conformity-report.json").exists()

                    if target_files:
                        has_dataset = True
                    if target_has_metrics:
                        has_metrics = True
                    if target_has_conformity:
                        has_conformity = True

                    targets.append({
                        "size": target_size,
                        "datasetFiles": target_files,
                        "hasMetrics": target_has_metrics,
                        "hasConformity": target_has_conformity,
                    })
            else:
                # Legacy flat structure: dataset/ or datasets/ with direct files
                flat_files = [f.name for f in _ds_dir.iterdir() if f.is_file()]
                dataset_files = flat_files
                has_dataset = bool(flat_files)

        # Legacy metrics dir (root level)
        if not has_metrics:
            legacy_metrics = job_dir / "metrics"
            has_metrics = legacy_metrics.exists() and any(legacy_metrics.iterdir())

        # Legacy conformity (reports/ dir)
        if not has_conformity:
            has_conformity = (job_dir / "reports" / "conformity-report.json").exists()

        has_tokens = (job_dir / "reports" / "tokens.json").exists()

        jobs.append({
            "dirName": job_dir.name,
            "path": str(job_dir),
            "jobId": job_id,
            "jobName": job_name,
            "hasDataset": has_dataset,
            "hasMavs": has_mavs,
            "hasMetrics": has_metrics,
            "hasConformity": has_conformity,
            "hasTokens": has_tokens,
            "datasetFiles": dataset_files,
            "mavModels": mav_models,
            "targets": targets,
        })

    return {"jobs": jobs}


class ReadTokensRequest(BaseModel):
    jobDir: str


@app.post("/api/read-tokens")
async def read_tokens(request: ReadTokensRequest):
    """Read tokens.json from a job's reports directory."""
    from pathlib import Path
    import json

    tokens_path = Path(request.jobDir) / "reports" / "tokens.json"
    if not tokens_path.exists():
        return {"found": False, "tokens": None}

    try:
        with open(tokens_path, "r", encoding="utf-8") as f:
            tokens_data = json.load(f)
        return {"found": True, "tokens": tokens_data}
    except Exception as e:
        return {"found": False, "tokens": None, "error": str(e)}


class FetchActualCostsRequest(BaseModel):
    generationIds: list[str]


@app.post("/api/fetch-actual-costs")
async def fetch_actual_costs(request: FetchActualCostsRequest):
    """Fetch actual costs from OpenRouter generation details endpoint."""
    import asyncio

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        return {"costs": {}, "error": "No API key configured"}

    valid_ids = [gid for gid in request.generationIds if gid]
    if not valid_ids:
        return {"costs": {}}

    costs: dict[str, float] = {}
    semaphore = asyncio.Semaphore(10)  # Limit concurrent requests

    async with httpx.AsyncClient() as client:
        async def fetch_one(gen_id: str):
            async with semaphore:
                try:
                    resp = await client.get(
                        f"https://openrouter.ai/api/v1/generation?id={gen_id}",
                        headers={"Authorization": f"Bearer {api_key}"},
                        timeout=10.0,
                    )
                    if resp.status_code == 200:
                        data = resp.json().get("data", {})
                        costs[gen_id] = data.get("usage", 0)
                except Exception:
                    pass

        await asyncio.gather(*[fetch_one(gid) for gid in valid_ids])

    return {"costs": costs}


class ReadDatasetRequest(BaseModel):
    jobDir: str
    filename: str


@app.post("/api/read-dataset")
async def read_dataset(request: ReadDatasetRequest):
    """Read and parse a dataset file from a job directory."""
    from pathlib import Path
    import json
    import xml.etree.ElementTree as ET

    # Check both datasets/ (new) and dataset/ (legacy)
    file_path = Path(request.jobDir) / "datasets" / request.filename
    if not file_path.exists():
        file_path = Path(request.jobDir) / "dataset" / request.filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {request.filename}")

    reviews = []

    if request.filename.endswith('.jsonl'):
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    reviews.append(json.loads(line))

    elif request.filename.endswith('.csv'):
        import csv
        # CSV is per-opinion rows, group by review_id
        review_map = {}
        with open(file_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rid = row.get("review_id", "0")
                if rid not in review_map:
                    review_map[rid] = {"id": rid, "sentences": {}, "metadata": {}}
                sid = row.get("sentence_id", f"{rid}:0")
                if sid not in review_map[rid]["sentences"]:
                    review_map[rid]["sentences"][sid] = {
                        "id": sid,
                        "text": row.get("text", ""),
                        "opinions": [],
                    }
                review_map[rid]["sentences"][sid]["opinions"].append({
                    "target": row.get("target", "NULL"),
                    "category": row.get("category", ""),
                    "polarity": row.get("polarity", ""),
                    "from": int(row.get("from", 0)),
                    "to": int(row.get("to", 0)),
                })
        for rid, rdata in review_map.items():
            reviews.append({
                "id": rid,
                "sentences": list(rdata["sentences"].values()),
                "metadata": rdata["metadata"],
            })

    elif request.filename.endswith('.xml'):
        tree = ET.parse(file_path)
        root = tree.getroot()
        for review_elem in root.findall('.//Review'):
            rid = review_elem.get('rid', '0')
            sentences = []
            for sent_elem in review_elem.findall('.//sentence'):
                sid = sent_elem.get('id', '')
                text_elem = sent_elem.find('text')
                text = text_elem.text if text_elem is not None else ""
                opinions = []
                for op_elem in sent_elem.findall('.//Opinion'):
                    # Support both standard (from/to) and hotel format (target_from/target_to)
                    from_val = op_elem.get('from') or op_elem.get('target_from') or '0'
                    to_val = op_elem.get('to') or op_elem.get('target_to') or '0'
                    opinions.append({
                        "target": op_elem.get('target', 'NULL'),
                        "category": op_elem.get('category', ''),
                        "polarity": op_elem.get('polarity', ''),
                        "from": int(from_val),
                        "to": int(to_val),
                    })
                sentences.append({"id": sid, "text": text, "opinions": opinions})
            reviews.append({"id": rid, "sentences": sentences, "metadata": {}})

    return {"reviews": reviews, "count": len(reviews), "format": file_path.suffix[1:]}


class ReadMavRequest(BaseModel):
    jobDir: str


@app.post("/api/read-mav-reports")
async def read_mav_reports(request: ReadMavRequest):
    """Read MAV reports and per-model data from a job directory."""
    from pathlib import Path
    import json

    mavs_dir = Path(request.jobDir) / "mavs"
    reports_dir = Path(request.jobDir) / "reports"

    # Read per-model data
    models_data = {}
    if mavs_dir.exists():
        for model_dir in sorted(mavs_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            model_name = model_dir.name
            model_data = {}

            understanding_path = model_dir / "understanding.md"
            if understanding_path.exists():
                model_data["understanding"] = understanding_path.read_text(encoding="utf-8")

            queries_path = model_dir / "queries.json"
            if queries_path.exists():
                model_data["queries"] = json.loads(queries_path.read_text(encoding="utf-8"))

            answers_path = model_dir / "answers.json"
            if answers_path.exists():
                model_data["answers"] = json.loads(answers_path.read_text(encoding="utf-8"))

            models_data[model_name] = model_data

    # Read MAV report (consensus data) - check mavs/ first (new), then reports/ (legacy)
    report = None
    for candidate_dir in [mavs_dir, reports_dir]:
        report_path = candidate_dir / "mav-report.json"
        if report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            break

    # Read summary CSV - same priority
    summary = None
    for candidate_dir in [mavs_dir, reports_dir]:
        summary_path = candidate_dir / "mav-summary.csv"
        if summary_path.exists():
            summary = summary_path.read_text(encoding="utf-8")
            break

    return {
        "models": models_data,
        "report": report,
        "summary": summary,
    }


class ScanTargetsRequest(BaseModel):
    jobDir: str


@app.post("/api/scan-targets")
async def scan_targets(request: ScanTargetsRequest):
    """Scan a job directory for available target sizes and their metrics status.

    Returns which targets exist under datasets/, whether each has MDQA metrics,
    and detected model slugs for multi-model jobs.
    """
    from pathlib import Path
    import json

    job_dir = Path(request.jobDir)
    datasets_dir = job_dir / "datasets"

    if not datasets_dir.exists():
        raise HTTPException(status_code=404, detail="No datasets directory found")

    targets = []
    is_multi_model = False

    for target_dir in sorted(datasets_dir.iterdir()):
        if not target_dir.is_dir() or not target_dir.name.isdigit():
            continue

        target_value = int(target_dir.name)
        metrics_dir = target_dir / "metrics"
        has_metrics = False
        metrics_files = []
        model_slugs = []

        if metrics_dir.exists():
            for fname in ("mdqa-results.json", "mdqa_metrics_average.json"):
                fpath = metrics_dir / fname
                if fpath.exists():
                    has_metrics = True
                    metrics_files.append(fname)
                    try:
                        data = json.loads(fpath.read_text(encoding="utf-8"))
                        if isinstance(data.get("perModel"), list) and len(data["perModel"]) > 1:
                            is_multi_model = True
                            model_slugs = [pm.get("modelSlug", "") for pm in data["perModel"] if pm.get("modelSlug")]
                    except Exception:
                        pass
                    break  # prefer mdqa-results.json if both exist

        targets.append({
            "targetValue": target_value,
            "countMode": "sentences",
            "hasMetrics": has_metrics,
            "metricsFiles": metrics_files,
            "modelSlugs": model_slugs,
        })

    return {
        "targets": targets,
        "isMultiModel": is_multi_model,
        "totalTargets": len(targets),
    }


class RerunEvaluationRequest(BaseModel):
    jobDir: str
    targetSize: Optional[int] = None  # If set, only re-evaluate this target


@app.post("/api/rerun-evaluation")
async def rerun_evaluation(request: RerunEvaluationRequest):
    """Re-run MDQA evaluation on existing generated datasets.

    Scans run directories, evaluates each dataset, and saves consolidated
    results in CERA format (runs array + average + std).
    Works for both CERA and heuristic jobs.
    """
    from pathlib import Path
    import json
    import csv as csv_mod
    import re as re_mod
    import numpy as np

    ds_dir = Path(request.jobDir) / "datasets"
    if not ds_dir.exists():
        raise HTTPException(status_code=404, detail="Datasets directory not found")

    # Determine which targets to evaluate
    target_dirs = sorted([
        d for d in ds_dir.iterdir()
        if d.is_dir() and d.name.isdigit()
    ], key=lambda d: int(d.name))

    if request.targetSize is not None:
        target_dirs = [d for d in target_dirs if d.name == str(request.targetSize)]
        if not target_dirs:
            raise HTTPException(status_code=404, detail=f"Target size {request.targetSize} not found")

    metric_keys = ["bleu", "rouge_l", "bertscore", "moverscore", "distinct_1", "distinct_2", "self_bleu"]
    metric_categories = {
        "lexical": ["bleu", "rouge_l"],
        "semantic": ["bertscore", "moverscore"],
        "diversity": ["distinct_1", "distinct_2", "self_bleu"],
    }
    metric_descriptions = {
        "bleu": "N-gram overlap precision score",
        "rouge_l": "Longest common subsequence recall",
        "bertscore": "Contextual similarity using BERT embeddings",
        "moverscore": "Earth mover distance on word embeddings",
        "distinct_1": "Unique unigram ratio",
        "distinct_2": "Unique bigram ratio",
        "self_bleu": "Intra-corpus similarity (lower = more diverse)",
    }

    # Build evaluation config with reference metrics enabled
    eval_config = {
        "metrics": ["bleu", "rouge_l", "bertscore", "moverscore", "distinct_1", "distinct_2", "self_bleu"],
        "reference_metrics_enabled": True,
    }

    results_by_target = {}

    for target_dir in target_dirs:
        target_size = int(target_dir.name)
        metrics_dir = target_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)

        # Find all explicit dataset files in run directories
        dataset_files = sorted(target_dir.glob("run*/**/*-explicit.jsonl"))
        if not dataset_files:
            # Try without -explicit suffix
            dataset_files = sorted(target_dir.glob("run*/**/*.jsonl"))
        if not dataset_files:
            continue

        # Evaluate each dataset file
        eval_metrics = []  # list of (metrics_dict, file_path, run_num, model_slug)
        job_paths = {
            "dataset": str(target_dir),
            "metrics": str(metrics_dir),
            "root": request.jobDir,
        }

        for dataset_file_path in dataset_files:
            try:
                metrics_result = await execute_evaluation(
                    job_id="rerun",
                    job_paths=job_paths,
                    evaluation_config=eval_config,
                    dataset_file=str(dataset_file_path),
                    convex=None,
                    save_files=False,
                )
                if metrics_result:
                    # Extract run number and model slug from path
                    fp = str(dataset_file_path)
                    run_match = re_mod.search(r'/run(\d+)/', fp)
                    run_num = int(run_match.group(1)) if run_match else 1
                    model_match = re_mod.search(r'/run\d+/([^/]+)/', fp)
                    model_slug = model_match.group(1) if model_match else "unknown"
                    clean_metrics = {k: float(round(float(v), 6)) for k, v in metrics_result.items()
                                    if k in metric_keys and v is not None}
                    eval_metrics.append((clean_metrics, fp, run_num, model_slug))
            except Exception as e:
                print(f"[Rerun] Warning: Failed to evaluate {dataset_file_path}: {e}")

        if not eval_metrics:
            continue

        # Group by run and average across models within each run
        per_run = {}
        for clean_m, _, rn, _ in eval_metrics:
            if rn not in per_run:
                per_run[rn] = []
            per_run[rn].append(clean_m)

        all_run_metrics = []
        for rn in sorted(per_run.keys()):
            run_avg = {}
            for key in metric_keys:
                vals = [m.get(key) for m in per_run[rn] if m.get(key) is not None]
                if vals:
                    run_avg[key] = float(round(sum(vals) / len(vals), 6))
            all_run_metrics.append(run_avg)

        # Compute average/std
        avg_metrics = {}
        std_metrics = {}
        for key in metric_keys:
            vals = [m.get(key) for m in all_run_metrics if m.get(key) is not None]
            if vals:
                arr = np.array([float(v) for v in vals])
                avg_metrics[key] = float(round(np.mean(arr), 6))
                std_metrics[key] = float(round(np.std(arr), 6)) if len(vals) > 1 else 0.0

        # Build per-model breakdown
        per_model_data = {}
        for clean_m, _, rn, m_slug in eval_metrics:
            if m_slug not in per_model_data:
                per_model_data[m_slug] = {"metrics_list": [], "per_run": []}
            per_model_data[m_slug]["metrics_list"].append(clean_m)
            per_model_data[m_slug]["per_run"].append({"run": rn, "metrics": clean_m})

        per_model_entries = []
        for m_slug, m_data in per_model_data.items():
            m_avg = {}
            for key in metric_keys:
                vals = [m.get(key) for m in m_data["metrics_list"] if m.get(key) is not None]
                if vals:
                    m_avg[key] = float(round(sum(vals) / len(vals), 6))
            per_model_entries.append({
                "model": m_slug, "modelSlug": m_slug,
                "metrics": m_avg if m_avg else None,
                "runs": sorted(m_data["per_run"], key=lambda x: x["run"]),
            })
        is_multi_model = len(per_model_entries) > 1

        # Save CSV per category
        for category, cat_metrics in metric_categories.items():
            csv_path = metrics_dir / f"{category}-metrics.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv_mod.writer(f)
                writer.writerow(["run", "metric", "score", "description"])
                for run_idx, run_met in enumerate(all_run_metrics, 1):
                    for metric in cat_metrics:
                        val = run_met.get(metric)
                        if val is not None:
                            writer.writerow([run_idx, metric, f"{val:.6f}", metric_descriptions.get(metric, "")])
                for metric in cat_metrics:
                    val = avg_metrics.get(metric)
                    if val is not None:
                        writer.writerow(["avg", metric, f"{val:.6f}", f"Average across {len(all_run_metrics)} runs"])
                for metric in cat_metrics:
                    val = std_metrics.get(metric)
                    if val is not None:
                        writer.writerow(["std", metric, f"{val:.6f}", "Standard deviation"])

        # Save consolidated JSON (CERA format)
        all_runs_json = {
            "runs": [],
            "average": {},
            "std": {},
            "totalRuns": len(all_run_metrics),
        }
        for run_idx, run_met in enumerate(all_run_metrics, 1):
            run_data = {"run": run_idx}
            for category, cat_metrics in metric_categories.items():
                run_data[category] = {m: run_met.get(m) for m in cat_metrics if run_met.get(m) is not None}
            all_runs_json["runs"].append(run_data)

        for category, cat_metrics in metric_categories.items():
            all_runs_json["average"][category] = {m: avg_metrics.get(m) for m in cat_metrics if avg_metrics.get(m) is not None}
            all_runs_json["std"][category] = {m: std_metrics.get(m) for m in cat_metrics}

        all_runs_json["average"]["_flat"] = avg_metrics
        all_runs_json["std"]["_flat"] = std_metrics

        if is_multi_model:
            all_runs_json["totalModels"] = len(per_model_entries)
            all_runs_json["perModel"] = per_model_entries

        json_path = metrics_dir / "mdqa-results.json"
        with open(json_path, "w") as f:
            json.dump(all_runs_json, f, indent=2)

        # Also save legacy average file
        agg_data = {}
        for key in metric_keys:
            if key in avg_metrics:
                agg_data[key] = avg_metrics[key]
            if key in std_metrics:
                agg_data[f"{key}_std"] = std_metrics[key]
        if is_multi_model:
            agg_data["totalModels"] = len(per_model_entries)
            agg_data["perModel"] = per_model_entries
        agg_path = metrics_dir / "mdqa_metrics_average.json"
        with open(agg_path, "w") as f:
            json.dump(agg_data, f, indent=2)

        results_by_target[target_size] = {
            "totalRuns": len(all_run_metrics),
            "totalDatasets": len(eval_metrics),
            "averageMetrics": avg_metrics,
        }
        print(f"[Rerun] Target {target_size}: Evaluated {len(eval_metrics)} datasets across {len(all_run_metrics)} runs")

    return {
        "success": True,
        "targets": results_by_target,
        "totalTargets": len(results_by_target),
    }


class ReadMetricsRequest(BaseModel):
    jobDir: str
    targetSize: Optional[int] = None  # For multi-target jobs: which target to read


@app.post("/api/read-metrics")
async def read_metrics(request: ReadMetricsRequest):
    """Read MDQA metrics from a job directory.

    Supports:
    - Single-run metrics: {category}-metrics.csv (no run column)
    - Multi-run metrics: {category}-metrics.csv (with run column)
    - Combined JSON: mdqa-results.json (with runs array for multi-run)
    - Per-target metrics: datasets/{targetSize}/metrics/
    """
    from pathlib import Path
    import csv
    import json

    # Check for per-target metrics first, then legacy root metrics
    metrics_dir = None
    if request.targetSize is not None:
        candidate = Path(request.jobDir) / "datasets" / str(request.targetSize) / "metrics"
        if candidate.exists():
            metrics_dir = candidate
    if metrics_dir is None:
        # Try datasets/*/metrics/ (auto-detect single target)
        ds_dir = Path(request.jobDir) / "datasets"
        if ds_dir.exists():
            target_dirs = [d for d in ds_dir.iterdir() if d.is_dir() and d.name.isdigit()]
            if len(target_dirs) == 1 and (target_dirs[0] / "metrics").exists():
                metrics_dir = target_dirs[0] / "metrics"
    if metrics_dir is None:
        metrics_dir = Path(request.jobDir) / "metrics"
    if not metrics_dir.exists():
        raise HTTPException(status_code=404, detail="Metrics directory not found")

    result = {}
    is_multi_run = False
    per_run_metrics = {}  # {category: {runN: [metrics]}}
    average_metrics = {}  # {category: [metrics]}
    std_metrics = {}  # {category: [metrics]}

    metric_descriptions = {
        "bleu": "N-gram overlap precision score",
        "rouge_l": "Longest common subsequence recall",
        "bertscore": "Contextual similarity using BERT embeddings",
        "moverscore": "Earth mover distance on word embeddings",
        "distinct_1": "Unique unigram ratio",
        "distinct_2": "Unique bigram ratio",
        "self_bleu": "Intra-corpus similarity (lower = more diverse)",
    }

    # Try to read from JSON first (preferred for multi-run)
    # Check both CERA format (mdqa-results.json) and heuristic format (mdqa_metrics_average.json)
    json_path = metrics_dir / "mdqa-results.json"
    heuristic_avg_path = metrics_dir / "mdqa_metrics_average.json"
    if not json_path.exists() and heuristic_avg_path.exists():
        # Heuristic format: flat {key: value, key_std: value} + per-run files
        import glob as glob_mod
        raw_avg = json.loads(heuristic_avg_path.read_text(encoding="utf-8"))

        # Read per-run files (mdqa_metrics-run1.json, mdqa_metrics-run2.json, etc.)
        run_files = sorted(metrics_dir.glob("mdqa_metrics-run*.json"))
        if run_files:
            is_multi_run = True
            category_map = {
                "bleu": "lexical", "rouge_l": "lexical",
                "bertscore": "semantic", "moverscore": "semantic",
                "distinct_1": "diversity", "distinct_2": "diversity", "self_bleu": "diversity",
            }
            for run_file in run_files:
                run_num = int(run_file.stem.split("run")[-1])
                run_data = json.loads(run_file.read_text(encoding="utf-8"))
                for metric, score in run_data.items():
                    if metric.endswith("_std") or metric.startswith("_"):
                        continue
                    cat = category_map.get(metric, "other")
                    if cat not in per_run_metrics:
                        per_run_metrics[cat] = {}
                    if f"run{run_num}" not in per_run_metrics[cat]:
                        per_run_metrics[cat][f"run{run_num}"] = []
                    per_run_metrics[cat][f"run{run_num}"].append(
                        {"metric": metric, "score": score, "description": metric_descriptions.get(metric, "")}
                    )

            # Build average and std from the average file
            for metric, score in raw_avg.items():
                if metric.endswith("_std") or metric.startswith("_"):
                    continue
                if metric in ("perModel", "totalModels"):
                    continue
                cat = category_map.get(metric, "other")
                if cat not in average_metrics:
                    average_metrics[cat] = []
                average_metrics[cat].append(
                    {"metric": metric, "score": score, "description": metric_descriptions.get(metric, "")}
                )
                std_val = raw_avg.get(f"{metric}_std")
                if std_val is not None:
                    if cat not in std_metrics:
                        std_metrics[cat] = []
                    std_metrics[cat].append(
                        {"metric": metric, "score": std_val, "description": "Standard deviation"}
                    )

            result["isMultiRun"] = True
            result["totalRuns"] = len(run_files)
            result["perRun"] = per_run_metrics
            result["average"] = average_metrics
            result["std"] = std_metrics
            for category in ["lexical", "semantic", "diversity"]:
                if category in average_metrics:
                    result[category] = average_metrics[category]

            # Extract per-model data (multi-model heuristic jobs)
            if "perModel" in raw_avg and isinstance(raw_avg["perModel"], list):
                result["perModel"] = raw_avg["perModel"]
                result["totalModels"] = raw_avg.get("totalModels", len(raw_avg["perModel"]))
        else:
            # Single average file, no per-run data
            category_map = {
                "bleu": "lexical", "rouge_l": "lexical",
                "bertscore": "semantic", "moverscore": "semantic",
                "distinct_1": "diversity", "distinct_2": "diversity", "self_bleu": "diversity",
            }
            for metric, score in raw_avg.items():
                if metric.endswith("_std") or metric.startswith("_"):
                    continue
                if metric in ("perModel", "totalModels"):
                    continue
                cat = category_map.get(metric, "other")
                if cat not in result:
                    result[cat] = []
                result[cat].append({"metric": metric, "score": score, "description": metric_descriptions.get(metric, "")})

            # Extract per-model data (multi-model heuristic jobs, single-run)
            if "perModel" in raw_avg and isinstance(raw_avg["perModel"], list):
                result["perModel"] = raw_avg["perModel"]
                result["totalModels"] = raw_avg.get("totalModels", len(raw_avg["perModel"]))

    if json_path.exists():
        raw = json.loads(json_path.read_text(encoding="utf-8"))

        # Check if it's the new multi-run format (has "runs" array)
        if "runs" in raw and isinstance(raw["runs"], list):
            is_multi_run = True
            # Extract per-run data
            for run_data in raw["runs"]:
                run_num = run_data.get("run", 1)
                for category in ["lexical", "semantic", "diversity"]:
                    if category in run_data:
                        if category not in per_run_metrics:
                            per_run_metrics[category] = {}
                        per_run_metrics[category][f"run{run_num}"] = [
                            {"metric": k, "score": v, "description": metric_descriptions.get(k, "")}
                            for k, v in run_data[category].items()
                            if v is not None
                        ]
            # Extract averages
            if "average" in raw:
                for category in ["lexical", "semantic", "diversity"]:
                    if category in raw["average"]:
                        average_metrics[category] = [
                            {"metric": k, "score": v, "description": metric_descriptions.get(k, "")}
                            for k, v in raw["average"][category].items()
                            if v is not None
                        ]
            # Extract std
            if "std" in raw:
                for category in ["lexical", "semantic", "diversity"]:
                    if category in raw["std"]:
                        std_metrics[category] = [
                            {"metric": k, "score": v, "description": "Standard deviation"}
                            for k, v in raw["std"][category].items()
                            if v is not None
                        ]

            result["isMultiRun"] = True
            result["totalRuns"] = raw.get("totalRuns", len(raw["runs"]))
            result["perRun"] = per_run_metrics
            result["average"] = average_metrics
            result["std"] = std_metrics
            # Set main metrics to averages for backward compatibility
            for category in ["lexical", "semantic", "diversity"]:
                if category in average_metrics:
                    result[category] = average_metrics[category]

            # Extract per-model data (multi-model jobs)
            if "perModel" in raw and isinstance(raw["perModel"], list):
                result["perModel"] = raw["perModel"]
                result["totalModels"] = raw.get("totalModels", len(raw["perModel"]))

        # Check if it's the grouped format (single-run with lexical/semantic/diversity keys)
        elif "lexical" in raw or "semantic" in raw or "diversity" in raw:
            for category in ["lexical", "semantic", "diversity"]:
                if category in raw and raw[category]:
                    result[category] = [
                        {"metric": k, "score": v, "description": metric_descriptions.get(k, "")}
                        for k, v in raw[category].items()
                        if v is not None
                    ]

            # If mdqa_metrics_average.json also exists, read std from it
            if heuristic_avg_path.exists():
                try:
                    avg_data = json.loads(heuristic_avg_path.read_text(encoding="utf-8"))
                    category_map = {
                        "bleu": "lexical", "rouge_l": "lexical",
                        "bertscore": "semantic", "moverscore": "semantic",
                        "distinct_1": "diversity", "distinct_2": "diversity", "self_bleu": "diversity",
                    }
                    for metric_key in category_map:
                        std_val = avg_data.get(f"{metric_key}_std")
                        if std_val is not None:
                            cat = category_map[metric_key]
                            if cat not in std_metrics:
                                std_metrics[cat] = []
                            std_metrics[cat].append(
                                {"metric": metric_key, "score": std_val, "description": "Standard deviation"}
                            )
                    if std_metrics:
                        result["std"] = std_metrics

                    # Count run directories to determine totalRuns
                    # (heuristic jobs have runN dirs but no per-run metric files)
                    target_dir = metrics_dir.parent  # datasets/{targetSize}/
                    run_dirs = sorted([
                        d for d in target_dir.iterdir()
                        if d.is_dir() and d.name.startswith("run") and d.name[3:].isdigit()
                    ])
                    if run_dirs:
                        result["totalRuns"] = len(run_dirs)
                except Exception:
                    pass

        # Old flat format
        else:
            category_map = {
                "bleu": "lexical", "rouge_l": "lexical",
                "bertscore": "semantic", "moverscore": "semantic",
                "distinct_1": "diversity", "distinct_2": "diversity", "self_bleu": "diversity",
            }
            for metric, score in raw.items():
                if metric.startswith("_"):  # Skip internal keys like _flat
                    continue
                cat = category_map.get(metric, "other")
                if cat not in result:
                    result[cat] = []
                result[cat].append({"metric": metric, "score": score, "description": metric_descriptions.get(metric, "")})

    # Fallback: read from CSVs if JSON not found or empty
    if not result:
        csv_per_model = {}  # model_slug -> {metric_key -> [values]}  (for multi-model CSVs)
        for category in ["lexical", "semantic", "diversity"]:
            csv_path = metrics_dir / f"{category}-metrics.csv"
            if csv_path.exists():
                metrics_by_run = {}
                avg_metrics = []
                std_row = []
                with open(csv_path, encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    has_model_col = False
                    for row in reader:
                        run_val = row.get("run", "")
                        model_val = row.get("model", "")
                        if model_val:
                            has_model_col = True
                        metric_entry = {
                            "metric": row.get("metric", ""),
                            "score": float(row.get("score", 0)),
                            "description": row.get("description", ""),
                        }
                        if run_val == "avg" and model_val and model_val != "ALL":
                            # Per-model average row
                            if model_val not in csv_per_model:
                                csv_per_model[model_val] = {}
                            csv_per_model[model_val][row.get("metric", "")] = float(row.get("score", 0))
                        elif run_val == "avg":
                            avg_metrics.append(metric_entry)
                        elif run_val == "std":
                            std_row.append(metric_entry)
                        elif run_val.isdigit():
                            # For multi-model CSVs with model col, only use ALL rows for per-run
                            if has_model_col and model_val and model_val != "ALL":
                                # Track per-model per-run data
                                pass
                            else:
                                run_key = f"run{run_val}"
                                if run_key not in metrics_by_run:
                                    metrics_by_run[run_key] = []
                                metrics_by_run[run_key].append(metric_entry)
                        else:
                            # Single-run format (no run column)
                            if category not in result:
                                result[category] = []
                            result[category].append(metric_entry)

                # If we found multi-run data
                if metrics_by_run:
                    is_multi_run = True
                    per_run_metrics[category] = metrics_by_run
                    if avg_metrics:
                        average_metrics[category] = avg_metrics
                        result[category] = avg_metrics  # Default to averages
                    if std_row:
                        std_metrics[category] = std_row

        if is_multi_run:
            result["isMultiRun"] = True
            result["totalRuns"] = len(list(per_run_metrics.values())[0]) if per_run_metrics else 0
            result["perRun"] = per_run_metrics
            result["average"] = average_metrics
            result["std"] = std_metrics

        # Build perModel from CSV per-model average rows
        if csv_per_model:
            result["perModel"] = [
                {"model": slug, "modelSlug": slug, "metrics": metrics}
                for slug, metrics in csv_per_model.items()
            ]
            result["totalModels"] = len(csv_per_model)

    return {"metrics": result}


# ─── LADy Results API ─────────────────────────────────────────────────────────

LADY_OUTPUT_DIR = os.environ.get("LADY_OUTPUT_DIR", "/app/lady-output")

# Mapping from LADy CSV metric names (@5 cutoff) to our canonical keys
_LADY_METRIC_MAP = {
    "P_5": "precision_at_5",
    "map_cut_5": "map_at_5",
    "ndcg_cut_5": "ndcg_at_5",
    "recall_5": "recall_at_5",
    "success_5": "specificity_at_5",
}


@app.post("/api/scan-lady-outputs")
async def scan_lady_outputs():
    """Scan the LADy output directory for available evaluation results.

    Returns a list of output directories, each with type and available target sizes.
    """
    from pathlib import Path

    output_root = Path(LADY_OUTPUT_DIR)
    if not output_root.exists():
        return {"outputs": []}

    outputs = []
    for d in sorted(output_root.iterdir()):
        if not d.is_dir():
            continue
        # Infer method type from directory name (e.g., "real", "cera", "heuristic", "real-1")
        base_name = d.name.split("-")[0] if "-" in d.name else d.name
        if base_name not in ("real", "cera", "heuristic"):
            continue
        # Discover target sizes
        targets = []
        for td in sorted(d.iterdir()):
            if td.is_dir() and td.name.startswith("target-"):
                size_str = td.name.replace("target-", "")
                if size_str.isdigit():
                    # Only include targets that have an aggregate.csv
                    if (td / "aggregate.csv").exists():
                        targets.append(int(size_str))
        if targets:
            outputs.append({
                "name": d.name,
                "path": str(d),
                "type": base_name,
                "targets": sorted(targets),
            })

    return {"outputs": outputs}


class ReadLadyMetricsRequest(BaseModel):
    outputDir: str
    targetSize: int


@app.post("/api/read-lady-metrics")
async def read_lady_metrics(request: ReadLadyMetricsRequest):
    """Read LADy evaluation metrics from an output directory for a specific target size.

    Reads aggregate.csv (cross-run mean ± std) and per-run agg.ad.pred.eval.mean.csv files.
    Returns metrics at @5 cutoff: P@5, MAP@5, NDCG@5, R@5, S@5.
    """
    from pathlib import Path
    import csv

    target_dir = Path(request.outputDir) / f"target-{request.targetSize}"
    agg_path = target_dir / "aggregate.csv"

    if not agg_path.exists():
        raise HTTPException(status_code=404, detail=f"No aggregate.csv found for target {request.targetSize}")

    # Parse aggregate.csv
    metrics = {}
    with open(agg_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metric_name = row.get("metric", "")
            canonical = _LADY_METRIC_MAP.get(metric_name)
            if canonical:
                mean_val = row.get("mean")
                std_val = row.get("std")
                if mean_val:
                    entry = {"mean": float(mean_val)}
                    if std_val and std_val.strip():
                        entry["std"] = float(std_val)
                    metrics[canonical] = entry

    # Parse per-run data from run directories
    per_run = []
    run_dirs = sorted(
        [d for d in target_dir.iterdir() if d.is_dir() and d.name.startswith("run")],
        key=lambda d: int(d.name.replace("run", "")) if d.name.replace("run", "").isdigit() else 0,
    )
    for run_dir in run_dirs:
        run_num_str = run_dir.name.replace("run", "")
        if not run_num_str.isdigit():
            continue
        run_num = int(run_num_str)
        run_csv = run_dir / "agg.ad.pred.eval.mean.csv"
        if not run_csv.exists():
            continue
        run_metrics = {}
        with open(run_csv, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                metric_name = row.get("metric", "")
                canonical = _LADY_METRIC_MAP.get(metric_name)
                if canonical:
                    mean_val = row.get("mean")
                    if mean_val:
                        run_metrics[canonical] = float(mean_val)
        if run_metrics:
            per_run.append({"run": run_num, "metrics": run_metrics})

    return {
        "metrics": metrics,
        "perRun": per_run if per_run else None,
        "nRuns": len(per_run) if per_run else (len(run_dirs) if run_dirs else 1),
    }


class ReadConformityRequest(BaseModel):
    jobDir: str
    targetSize: Optional[int] = None  # For multi-target jobs: which target to read


@app.post("/api/read-conformity")
async def read_conformity(request: ReadConformityRequest):
    """Read conformity report from a job directory.

    Returns conformity metrics for single-run or multi-run jobs.
    Multi-run jobs include per-run data, averages, and standard deviations.
    Checks per-target metrics/ first, then legacy reports/ dir.
    """
    from pathlib import Path
    import json

    report_path = None

    # Check per-target location first: datasets/{targetSize}/metrics/conformity-report.json
    if request.targetSize is not None:
        candidate = Path(request.jobDir) / "datasets" / str(request.targetSize) / "metrics" / "conformity-report.json"
        if candidate.exists():
            report_path = candidate

    # Auto-detect single target
    if report_path is None:
        ds_dir = Path(request.jobDir) / "datasets"
        if ds_dir.exists():
            target_dirs = [d for d in ds_dir.iterdir() if d.is_dir() and d.name.isdigit()]
            if len(target_dirs) == 1:
                candidate = target_dirs[0] / "metrics" / "conformity-report.json"
                if candidate.exists():
                    report_path = candidate

    # Legacy location: reports/conformity-report.json
    if report_path is None:
        candidate = Path(request.jobDir) / "reports" / "conformity-report.json"
        if candidate.exists():
            report_path = candidate

    if report_path is None:
        raise HTTPException(status_code=404, detail="Conformity report not found")

    raw = json.loads(report_path.read_text(encoding="utf-8"))
    return {"conformity": raw}


# =============================================================================
# Context Extraction API
# =============================================================================


class ExtractContextRequest(BaseModel):
    """Request model for context extraction from reference dataset."""
    reviews: list[str]  # Review texts to analyze
    extractSubject: bool = True
    extractReviewer: bool = True
    model: str  # Model to use for extraction (e.g., first MAV model)
    apiKey: str  # OpenRouter API key
    sampleCount: int = 25  # Number of reviews to sample (10-50, default 25)


class ExtractContextResponse(BaseModel):
    """Response model for context extraction."""
    subject_context: Optional[str] = None
    reviewer_context: Optional[str] = None
    sample_count: int = 0
    error: Optional[str] = None
    rde_usage: Optional[list[dict]] = None  # Serialized LLMUsage records for token tracking


@app.post("/api/extract-context", response_model=ExtractContextResponse)
async def extract_context(request: ExtractContextRequest):
    """
    Extract subject and/or reviewer context from reference dataset reviews.

    This endpoint uses LLM-based extraction to analyze a sample of reviews
    and infer contextual information about:
    - Subject: What type of product/service/entity is being reviewed
    - Reviewer: What demographics and characteristics typical reviewers have

    The extracted context can be used to domain-match generated reviews
    with real reference data for meaningful MDQA comparisons.
    """
    from cera.pipeline.context_extractor import ContextExtractor

    if not request.reviews:
        return ExtractContextResponse(error="No reviews provided")

    if not request.extractSubject and not request.extractReviewer:
        return ExtractContextResponse(error="At least one extraction type must be enabled")

    try:
        print(f"[CTX] Starting context extraction from {len(request.reviews)} reviews")
        print(f"[CTX] Using model: {request.model}")

        from cera.llm.usage import UsageTracker
        rde_tracker = UsageTracker()

        extractor = ContextExtractor(
            api_key=request.apiKey,
            model=request.model,
            usage_tracker=rde_tracker,
        )

        subject_context = None
        reviewer_context = None

        # Clamp sample count to valid range (10-50)
        sample_count = max(10, min(50, request.sampleCount))
        actual_sample_count = min(len(request.reviews), sample_count)
        print(f"[CTX] Sampling {actual_sample_count} reviews (requested: {sample_count})")

        # Extract based on request flags
        if request.extractSubject and request.extractReviewer:
            print("[CTX] Extracting subject context...")
            subject_context = await extractor.extract_subject_context(
                request.reviews, sample_count=sample_count
            )
            print(f"[CTX] Subject context extracted ({len(subject_context)} chars)")

            print("[CTX] Extracting reviewer context...")
            reviewer_context = await extractor.extract_reviewer_context(
                request.reviews, sample_count=sample_count
            )
            print(f"[CTX] Reviewer context extracted ({len(reviewer_context)} chars)")
        elif request.extractSubject:
            print("[CTX] Extracting subject context...")
            subject_context = await extractor.extract_subject_context(
                request.reviews, sample_count=sample_count
            )
            print(f"[CTX] Subject context extracted ({len(subject_context)} chars)")
        elif request.extractReviewer:
            print("[CTX] Extracting reviewer context...")
            reviewer_context = await extractor.extract_reviewer_context(
                request.reviews, sample_count=sample_count
            )
            print(f"[CTX] Reviewer context extracted ({len(reviewer_context)} chars)")

        print("[CTX] Context extraction complete")

        return ExtractContextResponse(
            subject_context=subject_context,
            reviewer_context=reviewer_context,
            sample_count=actual_sample_count,
            rde_usage=rde_tracker.records_list() if rde_tracker.total_tokens > 0 else None,
        )

    except Exception as e:
        return ExtractContextResponse(
            error=f"Context extraction failed: {str(e)}"
        )


# ============================================================================
# Reference Dataset Extraction (RDE) Endpoint
# ============================================================================

class DomainConfidence(BaseModel):
    """Domain inference with confidence score."""
    value: Optional[str] = None
    confidence: float = 0.0
    reason: Optional[str] = None


class RegionConfidence(BaseModel):
    """Region inference with confidence score."""
    value: Optional[str] = None
    confidence: float = 0.0
    reason: Optional[str] = None


class SexDistributionResult(BaseModel):
    """Sex distribution extracted from explicit text patterns."""
    male: float = 0.0
    female: float = 0.0
    unknown: float = 1.0
    detected_count: int = 0  # How many reviews had explicit sex mentions


class NoiseAnalysis(BaseModel):
    """Noise analysis results from reference dataset."""
    typo_rate: float = 0.0
    has_colloquialisms: bool = False
    sample_size: int = 0


class ReviewLengthStats(BaseModel):
    """Review length statistics."""
    avg_sentences: float = 0.0
    min_sentences: int = 0
    max_sentences: int = 0
    suggested_range: list[int] = [2, 5]


class PolarityDistribution(BaseModel):
    """Polarity distribution from reference dataset."""
    positive: float = 0.0
    neutral: float = 0.0
    negative: float = 0.0


class ExtractRefContextRequest(BaseModel):
    """Request model for comprehensive reference dataset extraction."""
    reviews: list[dict]  # Full review objects with sentences and opinions
    model: str  # LLM model for extraction
    apiKey: str  # OpenRouter API key
    sampleCount: int = 25  # Sample size for LLM extraction


class ExtractRefContextResponse(BaseModel):
    """Response model for comprehensive reference dataset extraction."""
    # LLM-extracted context
    subject_query: Optional[str] = None
    additional_context: Optional[str] = None
    reviewer_context: Optional[str] = None

    # Inferred values with confidence
    domain: Optional[DomainConfidence] = None
    region: Optional[RegionConfidence] = None

    # Statistics from reference dataset
    polarity: Optional[PolarityDistribution] = None
    sex_distribution: Optional[SexDistributionResult] = None
    noise: Optional[NoiseAnalysis] = None
    review_length: Optional[ReviewLengthStats] = None

    # Aspect categories
    aspect_categories: list[str] = []

    # Metadata
    sample_count: int = 0
    total_reviews: int = 0
    error: Optional[str] = None
    rde_usage: Optional[list[dict]] = None  # Serialized LLMUsage records for token tracking


def _load_sex_patterns() -> dict:
    """Load sex detection patterns from config file."""
    patterns_path = Path(__file__).parent / "config" / "sex-patterns.json"
    try:
        with open(patterns_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"[RDE] Warning: Could not load sex patterns: {e}")
        return {"male": [], "female": []}


def _detect_sex_from_text(text: str, patterns: dict) -> Optional[str]:
    """Detect sex from explicit patterns in text."""
    text_lower = text.lower()

    for pattern in patterns.get("male", []):
        if pattern in text_lower:
            return "male"

    for pattern in patterns.get("female", []):
        if pattern in text_lower:
            return "female"

    return None


def _analyze_sex_distribution(reviews: list[dict], patterns: dict) -> SexDistributionResult:
    """Analyze sex distribution from explicit text patterns."""
    male_count = 0
    female_count = 0
    unknown_count = 0

    for review in reviews:
        # Get all text from the review
        all_text = ""
        if "sentences" in review:
            for sentence in review["sentences"]:
                if isinstance(sentence, dict) and "text" in sentence:
                    all_text += " " + sentence["text"]
        elif "text" in review:
            all_text = review["text"]

        sex = _detect_sex_from_text(all_text, patterns)
        if sex == "male":
            male_count += 1
        elif sex == "female":
            female_count += 1
        else:
            unknown_count += 1

    total = male_count + female_count + unknown_count
    if total == 0:
        return SexDistributionResult()

    return SexDistributionResult(
        male=round(male_count / total, 2),
        female=round(female_count / total, 2),
        unknown=round(unknown_count / total, 2),
        detected_count=male_count + female_count,
    )


def _analyze_polarity_distribution(reviews: list[dict]) -> PolarityDistribution:
    """Analyze sentence-level polarity distribution.

    Counts individual opinion polarities across all sentences in all reviews,
    rather than aggregating to document-level via majority voting.
    This gives a more accurate representation of sentiment distribution.
    """
    positive_count = 0
    neutral_count = 0
    negative_count = 0

    for review in reviews:
        if "sentences" in review:
            for sentence in review["sentences"]:
                if isinstance(sentence, dict) and "opinions" in sentence:
                    for opinion in sentence["opinions"]:
                        polarity = opinion.get("polarity", "").lower()
                        if polarity == "positive":
                            positive_count += 1
                        elif polarity == "neutral":
                            neutral_count += 1
                        elif polarity == "negative":
                            negative_count += 1

    total = positive_count + neutral_count + negative_count
    if total == 0:
        return PolarityDistribution()

    return PolarityDistribution(
        positive=round(positive_count / total, 2),
        neutral=round(neutral_count / total, 2),
        negative=round(negative_count / total, 2),
    )


def _analyze_review_length(reviews: list[dict]) -> ReviewLengthStats:
    """Analyze review length statistics (sentences per review)."""
    sentence_counts = []

    for review in reviews:
        if "sentences" in review:
            sentence_counts.append(len(review["sentences"]))
        else:
            sentence_counts.append(1)

    if not sentence_counts:
        return ReviewLengthStats()

    avg = sum(sentence_counts) / len(sentence_counts)
    min_val = min(sentence_counts)
    max_val = max(sentence_counts)

    # Suggest a range based on the distribution
    suggested_min = max(1, int(avg - 1))
    suggested_max = min(10, int(avg + 2))

    return ReviewLengthStats(
        avg_sentences=round(avg, 1),
        min_sentences=min_val,
        max_sentences=max_val,
        suggested_range=[suggested_min, suggested_max],
    )


def _extract_aspect_categories(reviews: list[dict]) -> list[str]:
    """Extract unique aspect categories from reviews."""
    categories = set()

    for review in reviews:
        if "sentences" in review:
            for sentence in review["sentences"]:
                if isinstance(sentence, dict) and "opinions" in sentence:
                    for opinion in sentence["opinions"]:
                        if "category" in opinion and opinion["category"]:
                            categories.add(opinion["category"])

    return sorted(list(categories))


def _infer_domain_from_categories(categories: list[str]) -> DomainConfidence:
    """Infer domain from aspect categories."""
    if not categories:
        return DomainConfidence(reason="No categories found")

    category_str = " ".join(categories).upper()

    # Restaurant indicators
    restaurant_keywords = ["FOOD", "SERVICE", "AMBIANCE", "PRICE", "DRINKS", "LOCATION"]
    restaurant_score = sum(1 for kw in restaurant_keywords if kw in category_str)

    # Laptop/Electronics indicators
    laptop_keywords = ["LAPTOP", "DISPLAY", "SCREEN", "KEYBOARD", "BATTERY", "SUPPORT",
                       "OS", "SOFTWARE", "HARDWARE", "MEMORY", "GRAPHICS", "PORTS"]
    laptop_score = sum(1 for kw in laptop_keywords if kw in category_str)

    # Phone/Electronics indicators
    phone_keywords = ["PHONE", "CAMERA", "BATTERY", "SCREEN", "DISPLAY", "PERFORMANCE"]
    phone_score = sum(1 for kw in phone_keywords if kw in category_str)

    # Hotel indicators
    hotel_keywords = ["ROOM", "STAFF", "LOCATION", "CLEANLINESS", "VALUE", "AMENITIES"]
    hotel_score = sum(1 for kw in hotel_keywords if kw in category_str)

    scores = {
        "Restaurant": restaurant_score,
        "Laptop": laptop_score,
        "Electronics": phone_score,
        "Hotel": hotel_score,
    }

    if max(scores.values()) == 0:
        return DomainConfidence(value="General", confidence=0.3, reason="No domain-specific categories detected")

    best_domain = max(scores, key=scores.get)
    total_keywords = len(restaurant_keywords) + len(laptop_keywords) + len(phone_keywords) + len(hotel_keywords)
    confidence = min(1.0, scores[best_domain] / 5)  # Normalize to 0-1

    return DomainConfidence(
        value=best_domain,
        confidence=round(confidence, 2),
    )


def _infer_region_from_text(reviews: list[dict]) -> RegionConfidence:
    """Infer region from text patterns (currency, spellings, place names)."""
    all_text = ""
    for review in reviews:
        if "sentences" in review:
            for sentence in review["sentences"]:
                if isinstance(sentence, dict) and "text" in sentence:
                    all_text += " " + sentence["text"]

    all_text_lower = all_text.lower()

    # Currency indicators
    has_pound = "£" in all_text
    has_euro = "€" in all_text
    has_dollar = "$" in all_text

    # British English spellings
    british_spellings = ["colour", "favour", "honour", "flavour", "centre", "theatre", "metre"]
    british_count = sum(1 for word in british_spellings if word in all_text_lower)

    # American English spellings
    american_spellings = ["color", "favor", "honor", "flavor", "center", "theater", "meter"]
    american_count = sum(1 for word in american_spellings if word in all_text_lower)

    # Calculate scores
    scores = {
        "UK": (5 if has_pound else 0) + british_count * 2,
        "Europe": 5 if has_euro else 0,
        "US/Canada": (2 if has_dollar else 0) + american_count * 2,
    }

    max_score = max(scores.values())
    if max_score < 2:
        return RegionConfidence(
            value=None,
            confidence=0.0,
            reason="under 50% certainty threshold"
        )

    best_region = max(scores, key=scores.get)
    confidence = min(1.0, max_score / 10)

    if confidence < 0.5:
        return RegionConfidence(
            value=None,
            confidence=round(confidence, 2),
            reason="under 50% certainty threshold"
        )

    return RegionConfidence(
        value=best_region,
        confidence=round(confidence, 2),
    )


def _analyze_noise(reviews: list[dict], sample_size: int = 100) -> NoiseAnalysis:
    """Analyze noise level (typos, colloquialisms) in reviews."""
    try:
        from spellchecker import SpellChecker
        spell = SpellChecker()
    except ImportError:
        print("[RDE] Warning: pyspellchecker not installed, skipping noise analysis")
        return NoiseAnalysis(sample_size=0)

    # Colloquialism patterns
    colloquial_patterns = [
        r"\bgonna\b", r"\bwanna\b", r"\bgotta\b", r"\bkinda\b", r"\bsorta\b",
        r"\bcuz\b", r"\bcause\b", r"\btho\b", r"\bthough\b", r"\byeah\b",
        r"\bnope\b", r"\byup\b", r"\byep\b", r"\bbtw\b", r"\bimo\b",
        r"\bomg\b", r"\blol\b", r"\bbrb\b", r"\bidk\b", r"\bfyi\b",
    ]

    word_count = 0
    typo_count = 0
    has_colloquialisms = False
    sampled_reviews = reviews[:sample_size]

    for review in sampled_reviews:
        text = ""
        if "sentences" in review:
            for sentence in review["sentences"]:
                if isinstance(sentence, dict) and "text" in sentence:
                    text += " " + sentence["text"]
        elif "text" in review:
            text = review["text"]

        # Check for colloquialisms
        if not has_colloquialisms:
            for pattern in colloquial_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    has_colloquialisms = True
                    break

        # Count typos
        words = re.findall(r'\b[a-zA-Z]+\b', text)
        word_count += len(words)
        misspelled = spell.unknown(words)
        typo_count += len(misspelled)

    typo_rate = typo_count / word_count if word_count > 0 else 0

    return NoiseAnalysis(
        typo_rate=round(typo_rate, 4),
        has_colloquialisms=has_colloquialisms,
        sample_size=len(sampled_reviews),
    )


@app.post("/api/extract-ref-context", response_model=ExtractRefContextResponse)
async def extract_ref_context(request: ExtractRefContextRequest):
    """
    Comprehensive reference dataset extraction for auto-filling COMPOSITION settings.

    This endpoint analyzes a reference dataset to extract:
    - Subject query and additional context (LLM-generated)
    - Domain and region inference with confidence scores
    - Polarity distribution
    - Sex distribution (from explicit text patterns)
    - Noise analysis (typo rate, colloquialisms)
    - Aspect categories
    - Review length statistics

    The extracted data is used to auto-fill the COMPOSITION step with appropriate values.
    """
    from cera.pipeline.context_extractor import ContextExtractor

    if not request.reviews:
        return ExtractRefContextResponse(error="No reviews provided")

    try:
        print(f"[RDE] Starting reference dataset extraction from {len(request.reviews)} reviews")
        print(f"[RDE] Using model: {request.model}")

        total_reviews = len(request.reviews)

        # Load sex patterns
        sex_patterns = _load_sex_patterns()

        # 1. Extract aspect categories (no LLM needed)
        print("[RDE] Step 1/6: Extracting aspect categories...")
        aspect_categories = _extract_aspect_categories(request.reviews)
        print(f"[RDE] Found {len(aspect_categories)} unique categories")

        # 2. Analyze polarity distribution (no LLM needed)
        print("[RDE] Step 2/6: Analyzing polarity distribution...")
        polarity = _analyze_polarity_distribution(request.reviews)
        print(f"[RDE] Polarity: +{polarity.positive} n{polarity.neutral} -{polarity.negative}")

        # 3. Analyze sex distribution (no LLM needed)
        print("[RDE] Step 3/6: Analyzing sex distribution...")
        sex_distribution = _analyze_sex_distribution(request.reviews, sex_patterns)
        print(f"[RDE] Sex: m{sex_distribution.male} f{sex_distribution.female} u{sex_distribution.unknown}")

        # 4. Analyze noise (no LLM needed)
        print("[RDE] Step 4/6: Analyzing noise levels...")
        noise = _analyze_noise(request.reviews, sample_size=100)
        print(f"[RDE] Noise: {noise.typo_rate:.2%} typos, colloquialisms={noise.has_colloquialisms}")

        # 5. Analyze review length (no LLM needed)
        print("[RDE] Step 5/6: Analyzing review lengths...")
        review_length = _analyze_review_length(request.reviews)
        print(f"[RDE] Length: avg {review_length.avg_sentences} sentences")

        # 6. LLM-based extraction for subject context, reviewer context, domain, region, and query
        print("[RDE] Step 6/6: LLM extraction...")
        from cera.llm.usage import UsageTracker
        rde_tracker = UsageTracker()

        extractor = ContextExtractor(
            api_key=request.apiKey,
            model=request.model,
            usage_tracker=rde_tracker,
        )

        # Convert reviews to text strings for the extractor
        review_texts = []
        for review in request.reviews:
            if "sentences" in review:
                text = " ".join(
                    s["text"] for s in review["sentences"]
                    if isinstance(s, dict) and "text" in s
                )
                if text:
                    review_texts.append(text)

        sample_count = max(10, min(50, request.sampleCount))
        actual_sample = min(len(review_texts), sample_count)

        subject_context = await extractor.extract_subject_context(
            review_texts, sample_count=sample_count
        )
        print(f"[RDE] Subject context extracted ({len(subject_context)} chars)")

        reviewer_context = await extractor.extract_reviewer_context(
            review_texts, sample_count=sample_count
        )
        print(f"[RDE] Reviewer context extracted ({len(reviewer_context)} chars)")

        # Generate a concise subject query via LLM
        subject_query = await extractor.extract_subject_query(
            review_texts, sample_count=sample_count
        )
        print(f"[RDE] Subject query extracted: {subject_query}")

        # Extract domain via LLM
        domain_result = await extractor.extract_domain(
            review_texts, sample_count=sample_count
        )
        domain = DomainConfidence(
            value=domain_result.get("value"),
            confidence=domain_result.get("confidence", 0.0),
        )
        print(f"[RDE] Domain extracted: {domain.value} ({domain.confidence:.0%})")

        # Extract region via LLM
        region_result = await extractor.extract_region(
            review_texts, sample_count=sample_count
        )
        region = RegionConfidence(
            value=region_result.get("value"),
            confidence=region_result.get("confidence", 0.0),
            reason=region_result.get("reason"),
        )
        print(f"[RDE] Region extracted: {region.value or 'Unknown'} ({region.confidence:.0%})")

        print("[RDE] Reference dataset extraction complete")

        return ExtractRefContextResponse(
            subject_query=subject_query,
            additional_context=subject_context,
            reviewer_context=reviewer_context,
            domain=domain,
            region=region,
            polarity=polarity,
            sex_distribution=sex_distribution,
            noise=noise,
            review_length=review_length,
            aspect_categories=aspect_categories,
            sample_count=actual_sample,
            total_reviews=total_reviews,
            rde_usage=rde_tracker.records_list() if rde_tracker.total_tokens > 0 else None,
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return ExtractRefContextResponse(
            error=f"Reference dataset extraction failed: {str(e)}"
        )


# ============================================================
# Domain Pattern Management Endpoints
# ============================================================

class DomainPattern(BaseModel):
    """A domain pattern for Quick Stats inference."""
    name: str
    keywords: list[str]


class DomainPatternsResponse(BaseModel):
    """Response containing all domain patterns."""
    domains: list[DomainPattern]


class UpdateDomainPatternsRequest(BaseModel):
    """Request to update domain patterns."""
    domains: list[DomainPattern]


def _get_domain_patterns_path() -> Path:
    """Get the path to the domain patterns file."""
    return Path(__file__).parent / "config" / "domain-patterns.json"


def _get_domain_patterns_defaults_path() -> Path:
    """Get the path to the default domain patterns file."""
    return Path(__file__).parent / "config" / "domain-patterns-defaults.json"


@app.get("/api/domain-patterns")
async def get_domain_patterns() -> DomainPatternsResponse:
    """Get all domain patterns."""
    patterns_path = _get_domain_patterns_path()

    if not patterns_path.exists():
        # Fall back to defaults if main file doesn't exist
        patterns_path = _get_domain_patterns_defaults_path()

    try:
        with open(patterns_path, "r") as f:
            data = json.load(f)
            return DomainPatternsResponse(domains=data.get("domains", []))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read domain patterns: {str(e)}")


@app.put("/api/domain-patterns")
async def update_domain_patterns(request: UpdateDomainPatternsRequest) -> DomainPatternsResponse:
    """Update domain patterns (replaces all patterns)."""
    patterns_path = _get_domain_patterns_path()

    try:
        data = {
            "_comment": "Domain inference patterns for Quick Stats. Each domain has keywords to match against aspect categories. This file can be edited via the UI.",
            "domains": [{"name": d.name, "keywords": d.keywords} for d in request.domains]
        }
        with open(patterns_path, "w") as f:
            json.dump(data, f, indent=2)

        return DomainPatternsResponse(domains=request.domains)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update domain patterns: {str(e)}")


@app.post("/api/domain-patterns/reset")
async def reset_domain_patterns() -> DomainPatternsResponse:
    """Reset domain patterns to defaults."""
    defaults_path = _get_domain_patterns_defaults_path()
    patterns_path = _get_domain_patterns_path()

    try:
        # Read defaults
        with open(defaults_path, "r") as f:
            defaults_data = json.load(f)

        # Write to main file
        data = {
            "_comment": "Domain inference patterns for Quick Stats. Each domain has keywords to match against aspect categories. This file can be edited via the UI.",
            "domains": defaults_data.get("domains", [])
        }
        with open(patterns_path, "w") as f:
            json.dump(data, f, indent=2)

        return DomainPatternsResponse(domains=defaults_data.get("domains", []))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reset domain patterns: {str(e)}")
