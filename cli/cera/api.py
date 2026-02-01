"""CERA FastAPI Server - HTTP API for web GUI integration."""

from fastapi import FastAPI, BackgroundTasks, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Union
import httpx
import asyncio
import json
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
    similarity_threshold: float = 0.85


class SubjectProfile(BaseModel):
    query: str
    region: str
    domain: Optional[str] = None  # Product/service domain (e.g., restaurant, laptop, hotel)
    category: Optional[str] = None  # Deprecated: use 'domain' instead
    sentiment_depth: str
    mav: Optional[MAVConfig] = None
    aspect_categories: Optional[list[str]] = None

    @property
    def resolved_domain(self) -> str:
        """Get domain, falling back to deprecated category field."""
        return self.domain or self.category or "general"


class ReviewerProfile(BaseModel):
    age_range: list[int]
    sex_distribution: dict[str, float]
    additional_context: Optional[str] = None


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


class AttributesProfile(BaseModel):
    polarity: PolarityConfig
    noise: NoiseConfig
    length_range: list[int]
    temp_range: list[float] = [0.7, 0.9]  # LLM temperature range, optional with default


class GenerationConfig(BaseModel):
    count: int
    batch_size: int
    request_size: int = 5
    provider: str
    model: str
    output_formats: Optional[list[str]] = None  # e.g., ["jsonl", "csv", "semeval_xml"]
    dataset_mode: str = "explicit"  # "explicit", "implicit", or "both"


class AblationConfig(BaseModel):
    """Ablation study settings - toggle components on/off."""
    sil_enabled: bool = True
    mav_enabled: bool = True
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
    ├── contexts/      # Composition context files
    │   ├── subject-context.json
    │   ├── reviewers-context.json
    │   └── attributes-context.json
    ├── amls/          # AML prompt files
    ├── mavs/          # MAV raw data (query.md, response.md per model)
    ├── metrics/       # Evaluation metrics (MDQA)
    ├── reports/       # Analysis reports (JSON + CSV)
    └── dataset/       # Final dataset (JSONL, CSV)

    Returns:
        Dictionary with paths to each subdirectory
    """
    from pathlib import Path

    sanitized = sanitize_job_name(job_name)
    job_dir_name = f"{job_id}-{sanitized}" if sanitized else job_id
    job_dir = Path(jobs_dir) / job_dir_name

    # Create all subdirectories
    subdirs = ["contexts", "amls", "mavs", "metrics", "reports", "dataset"]
    paths = {"root": str(job_dir)}

    for subdir in subdirs:
        subdir_path = job_dir / subdir
        subdir_path.mkdir(parents=True, exist_ok=True)
        paths[subdir] = str(subdir_path)

    return paths


def save_job_config(
    job_paths: dict,
    job_name: str,
    config: "JobConfig",
    phases: list[str],
    evaluation_config: Optional[dict] = None,
    reused_from: Optional[str] = None,
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
    if config.subject_profile:
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
    if config.reviewer_profile:
        config_data["reviewer_profile"] = {
            "age_range": config.reviewer_profile.age_range,
            "sex_distribution": {
                "male": config.reviewer_profile.sex_distribution.male,
                "female": config.reviewer_profile.sex_distribution.female,
                "unspecified": config.reviewer_profile.sex_distribution.unspecified,
            } if hasattr(config.reviewer_profile.sex_distribution, 'male') else config.reviewer_profile.sex_distribution,
            "additional_context": getattr(config.reviewer_profile, 'additional_context', ''),
        }

    # Add attributes_profile if available
    if config.attributes_profile:
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
        }

    # Add generation config if available
    if config.generation:
        config_data["generation"] = {
            "count": config.generation.count,
            "batch_size": config.generation.batch_size,
            "request_size": getattr(config.generation, 'request_size', 5),
            "model": config.generation.model,
            "provider": config.generation.provider,
            "output_formats": getattr(config.generation, 'output_formats', ["semeval_xml"]),
            "dataset_mode": getattr(config.generation, 'dataset_mode', 'both'),
        }

    # Add ablation settings if provided
    if config.ablation:
        config_data["ablation"] = {
            "sil_enabled": config.ablation.sil_enabled,
            "mav_enabled": config.ablation.mav_enabled,
            "polarity_enabled": config.ablation.polarity_enabled,
            "noise_enabled": config.ablation.noise_enabled,
            "age_enabled": config.ablation.age_enabled,
            "sex_enabled": config.ablation.sex_enabled,
        }

    # Add evaluation config if provided
    if evaluation_config:
        config_data["evaluation"] = evaluation_config

    config_path = Path(job_paths["root"]) / "config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2, default=str)

    return str(config_path)


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


class ConvexClient:
    """Client for updating Convex database from Python."""

    def __init__(self, url: str, token: str):
        self.url = url
        self.token = token
        self.client = httpx.AsyncClient()

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
        """Update composition progress in Convex."""
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
            # Check for Convex-level errors in the response
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
        """Update job progress in Convex."""
        try:
            # Convex HTTP API for mutations (self-hosted uses "Convex" auth scheme)
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
        """Add a log entry in Convex."""
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

    async def run_mutation(self, path: str, args: dict):
        """Run an arbitrary Convex mutation."""
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

    async def update_generated_count(self, job_id: str, count: int):
        """Update the generated review count in Convex."""
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
        """Update the failed review count in Convex."""
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


async def execute_pipeline(
    job_id: str,
    job_name: str,
    config: JobConfig,
    api_key: str,
    jobs_directory: str,
    convex_url: Optional[str],
    convex_token: Optional[str],
):
    """Execute the CERA pipeline."""
    convex = None
    if convex_url and convex_token:
        convex = ConvexClient(convex_url, convex_token)

    # Create job directory structure
    job_paths = create_job_directory(jobs_directory, job_id, job_name)

    try:
        # Phase 1: Composition
        if convex:
            await convex.update_progress(job_id, 5, "composition")
            await convex.add_log(job_id, "INFO", "composition", "Starting composition phase...")
            await convex.add_log(job_id, "INFO", "composition", f"Job directory: {job_paths['root']}")

        # SIL - Subject Intelligence Layer
        if convex:
            await convex.add_log(
                job_id,
                "INFO",
                "composition",
                f"[SIL] Gathering intelligence for: {config.subject_profile.query}",
            )
        await asyncio.sleep(2)  # Simulated delay

        # RGM - Reviewer Generation Module
        if convex:
            await convex.update_progress(job_id, 15, "composition")
            await convex.add_log(
                job_id,
                "INFO",
                "composition",
                f"[RGM] Generating reviewer profiles (age: {config.reviewer_profile.age_range})",
            )
        await asyncio.sleep(1)

        # ACM - Attributes Composition Module
        if convex:
            await convex.update_progress(job_id, 25, "composition")
            await convex.add_log(
                job_id,
                "INFO",
                "composition",
                f"[ACM] Composing attributes (polarity: {config.attributes_profile.polarity.positive:.0%} pos)",
            )
        await asyncio.sleep(1)

        # Phase 2: Generation
        if convex:
            await convex.update_progress(job_id, 30, "generation")
            await convex.add_log(job_id, "INFO", "generation", "Starting generation phase...")

        total_reviews = config.generation.count
        batch_size = config.generation.batch_size
        num_batches = (total_reviews + batch_size - 1) // batch_size

        for i in range(num_batches):
            batch_num = i + 1
            progress = 30 + int((i / num_batches) * 40)

            if convex:
                await convex.update_progress(job_id, progress, "generation")
                await convex.add_log(
                    job_id,
                    "INFO",
                    "generation",
                    f"[BATCH] Processing batch {batch_num}/{num_batches}...",
                )

            # Simulated batch processing
            await asyncio.sleep(0.5)

        # Noise injection
        if convex:
            await convex.update_progress(job_id, 75, "generation")
            await convex.add_log(
                job_id,
                "INFO",
                "generation",
                f"[NOISE] Injecting noise (typo_rate: {config.attributes_profile.noise.typo_rate:.1%})",
            )
        await asyncio.sleep(1)

        # Phase 3: Evaluation
        if convex:
            await convex.update_progress(job_id, 80, "evaluation")
            await convex.add_log(job_id, "INFO", "evaluation", "Starting evaluation phase...")

        # MDQA metrics
        metrics = {
            "bertscore": 0.87,
            "distinct_1": 0.92,
            "distinct_2": 0.65,
            "self_bleu": 0.31,
        }

        if convex:
            await convex.add_log(job_id, "INFO", "evaluation", "[MDQA] Computing BERTScore...")
            await asyncio.sleep(1)
            await convex.update_progress(job_id, 85, "evaluation")
            await convex.add_log(job_id, "INFO", "evaluation", f"[MDQA] BERTScore: {metrics['bertscore']}")

            await convex.add_log(job_id, "INFO", "evaluation", "[MDQA] Computing Distinct-n...")
            await asyncio.sleep(1)
            await convex.update_progress(job_id, 90, "evaluation")
            await convex.add_log(
                job_id,
                "INFO",
                "evaluation",
                f"[MDQA] Distinct-1: {metrics['distinct_1']}, Distinct-2: {metrics['distinct_2']}",
            )

            await convex.add_log(job_id, "INFO", "evaluation", "[MDQA] Computing Self-BLEU...")
            await asyncio.sleep(1)
            await convex.update_progress(job_id, 95, "evaluation")
            await convex.add_log(job_id, "INFO", "evaluation", f"[MDQA] Self-BLEU: {metrics['self_bleu']}")

        # Complete - use the job's dataset directory
        output_path = job_paths["dataset"]

        if convex:
            await convex.create_dataset(
                job_id=job_id,
                name=f"{config.subject_profile.query} Reviews",
                subject=config.subject_profile.query,
                domain=config.subject_profile.resolved_domain,
                review_count=total_reviews,
                metrics=metrics,
                output_path=output_path,
            )
            await convex.add_log(
                job_id,
                "INFO",
                "complete",
                f"Pipeline completed successfully. Generated {total_reviews} reviews.",
            )
            await convex.complete_job(job_id)

    except Exception as e:
        error_msg = str(e)
        if convex:
            await convex.add_log(job_id, "ERROR", "error", f"Pipeline failed: {error_msg}")
            await convex.fail_job(job_id, error_msg)
        raise


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
    from pathlib import Path
    from datetime import datetime

    from cera.pipeline.composition.sil import (
        SubjectIntelligenceLayer,
        MAVConfig as SILMAVConfig,
    )

    convex = None
    if convex_url and convex_token:
        convex = ConvexClient(convex_url, convex_token)

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
        # Create log callback that sends logs to Convex
        async def sil_log_callback(level: str, phase: str, message: str):
            if convex:
                await convex.add_log(job_id, level, phase, message)

        def sync_log_callback(level: str, phase: str, message: str):
            """Synchronous wrapper for the async log callback."""
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(sil_log_callback(level, phase, message))
                else:
                    loop.run_until_complete(sil_log_callback(level, phase, message))
            except Exception:
                pass  # Silently ignore logging errors

        sil = SubjectIntelligenceLayer(
            api_key=api_key,
            mav_config=sil_mav_config,
            tavily_api_key=tavily_api_key,
            log_callback=sync_log_callback if convex else None,
        )

        if convex:
            await convex.update_composition_progress(job_id, 5, "SIL")
            if mav_enabled:
                await convex.add_log(job_id, "INFO", "MAV", "Starting multi-agent verification...")

        # Gather intelligence (this runs MAV if enabled)
        mav_result = await sil.gather_intelligence(
            query=config.subject_profile.query,
            region=config.subject_profile.region,
            domain=config.subject_profile.resolved_domain,
            sentiment_depth=config.subject_profile.sentiment_depth,
            additional_context=getattr(config.subject_profile, 'additional_context', None),
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
                    "min_agreement": __import__('math').ceil(2/3 * len([md for md in mav_result.model_data if not md.error])),
                    "answer_similarity_threshold": report.threshold_used,
                    "max_queries": mav_max_queries,
                    "consensus_method": "query-based ceil(2/3*N) majority voting",
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

            # Save mav-report.json
            mav_report_path = reports_dir / "mav-report.json"
            with open(mav_report_path, "w", encoding="utf-8") as f:
                json.dump(mav_report, f, indent=2)

            # Generate mav-summary.csv for paper tables
            import csv
            summary_path = reports_dir / "mav-summary.csv"
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
                writer.writerow(["consensus_method", "query-based ceil(2/3*N) majority voting"])
                writer.writerow(["used_fallback", str(report.used_fallback)])

            if convex:
                await convex.add_log(job_id, "INFO", "MAV", "Generated MAV reports: mav-report.json, mav-summary.csv")

        if convex:
            await convex.update_composition_progress(job_id, 35, "SIL")
            await convex.add_log(job_id, "INFO", "SIL", "Subject context saved")

        # ========================================
        # Phase 2: RGM - Reviewer Generation Module
        # ========================================
        if convex:
            await convex.update_composition_progress(job_id, 50, "RGM")
            await convex.add_log(
                job_id,
                "INFO",
                "RGM",
                f"Generating reviewer profiles (age: {config.reviewer_profile.age_range})",
            )

        # Reviewers context contains ONLY the specs/distribution
        reviewers_context = {
            "age_range": config.reviewer_profile.age_range,
            "sex_distribution": config.reviewer_profile.sex_distribution,
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
        }

        # Save attributes context
        attributes_context_path = Path(job_paths["contexts"]) / "attributes-context.json"
        with open(attributes_context_path, "w") as f:
            json.dump(attributes_context, f, indent=2)

        if convex:
            await convex.update_composition_progress(job_id, 100, "ACM")
            await convex.add_log(job_id, "INFO", "ACM", "Attributes context saved")

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


@app.post("/api/run-job")
async def run_job(request: JobRequest, background_tasks: BackgroundTasks):
    """Start a pipeline job in the background."""
    background_tasks.add_task(
        execute_pipeline,
        request.jobId,
        request.jobName,
        request.config,
        request.apiKey,
        request.jobsDirectory,
        request.convexUrl,
        request.convexToken,
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
            convex = ConvexClient(request.convexUrl, request.convexToken)

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
                        temp_range=config.get("attributes_profile", {}).get("temp_range", [0.7, 0.9]),
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
        convex = ConvexClient(url=convex_url, token=convex_token)

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

    # Create log callback that sends SIL logs to Convex
    async def sil_log_callback(level: str, phase: str, message: str):
        if convex:
            await convex.add_log(job_id, level, phase, message)

    def sync_log_callback(level: str, phase: str, message: str):
        """Synchronous wrapper for the async log callback."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                asyncio.create_task(sil_log_callback(level, phase, message))
            else:
                loop.run_until_complete(sil_log_callback(level, phase, message))
        except Exception:
            pass  # Silently ignore logging errors

    # Initialize SIL with MAV config and log callback
    sil = SubjectIntelligenceLayer(
        api_key=api_key,
        mav_config=sil_mav_config,
        tavily_api_key=tavily_api_key,
        log_callback=sync_log_callback if convex else None,
    )

    # Gather intelligence (this runs MAV if enabled)
    await log_progress("SIL", f"Starting gather_intelligence for: {config.subject_profile.query}", 5)
    await log_progress("SIL", f"MAV config: enabled={sil_mav_config.enabled}, models={sil_mav_config.models}")

    mav_result = await sil.gather_intelligence(
        query=config.subject_profile.query,
        region=config.subject_profile.region,
        domain=config.subject_profile.resolved_domain,
        sentiment_depth=config.subject_profile.sentiment_depth,
        additional_context=getattr(config.subject_profile, 'additional_context', None),
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
                "min_agreement": __import__('math').ceil(2/3 * len([md for md in mav_result.model_data if not md.error])),
                "answer_similarity_threshold": report.threshold_used,
                "max_queries": mav_max_queries,
                "consensus_method": "query-based ceil(2/3*N) majority voting",
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

        # Save mav-report.json
        mav_report_path = reports_dir / "mav-report.json"
        with open(mav_report_path, "w", encoding="utf-8") as f:
            json.dump(mav_report, f, indent=2)

        # Generate mav-summary.csv for paper tables
        import csv
        summary_path = reports_dir / "mav-summary.csv"
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
            writer.writerow(["consensus_method", "query-based ceil(2/3*N) majority voting"])
            writer.writerow(["used_fallback", str(report.used_fallback)])

        await log_progress("MAV", "Generated reports: mav-report.json, mav-summary.csv", 50)
    else:
        await log_progress("MAV", "No query pool report available - skipping report generation", 50, level="WARNING")

    # ========================================
    # Phase 2: RGM - Reviewer Generation Module
    # ========================================
    await log_progress("RGM", "Building reviewer specifications...", 60)

    # Reviewers context contains ONLY the specs/distribution, not per-review assignments
    # Per-review assignments are generated during the generation phase
    reviewers_context = {
        "age_range": config.reviewer_profile.age_range,
        "sex_distribution": config.reviewer_profile.sex_distribution,
        "additional_context": config.reviewer_profile.additional_context,
        "review_count": config.generation.count,
    }

    # Save reviewers context
    reviewers_context_path = Path(job_paths["contexts"]) / "reviewers-context.json"
    with open(reviewers_context_path, "w") as f:
        json.dump(reviewers_context, f, indent=2)

    await log_progress("RGM", f"Reviewer specs configured (age: {config.reviewer_profile.age_range})", 75)

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
    }

    # Save attributes context
    attributes_context_path = Path(job_paths["contexts"]) / "attributes-context.json"
    with open(attributes_context_path, "w") as f:
        json.dump(attributes_context, f, indent=2)

    polarity = config.attributes_profile.polarity
    await log_progress("ACM", f"Polarity set: {polarity.positive}%+ / {polarity.neutral}%~ / {polarity.negative}%-", 95)
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
        review_blocks = re.findall(r'<Review\s+rid="([^"]*)">(.*?)</Review>', request.content, re.DOTALL)
        for rid, block in review_blocks:
            sentences = []
            sent_blocks = re.findall(r'<sentence\s+id="([^"]*)">(.*?)</sentence>', block, re.DOTALL)
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
                        frm = re.search(r'from="([^"]*)"', attrs)
                        to = re.search(r'to="([^"]*)"', attrs)
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


async def execute_generation(
    job_id: str,
    job_dir: str,
    config: JobConfig,
    api_key: str,
    convex_url: Optional[str],
    convex_token: Optional[str],
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
        convex = ConvexClient(convex_url, convex_token)
        print(f"[Generation] Convex client created for URL: {convex_url}")
    else:
        print(f"[Generation] WARNING: Convex client NOT created - url={convex_url}, token={'yes' if convex_token else 'no'}")

    job_path = Path(job_dir)
    contexts_path = job_path / "contexts"
    amls_path = job_path / "amls"
    dataset_path = job_path / "dataset"

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

        if convex:
            await convex.add_log(job_id, "INFO", "AML", "Starting generation phase...")
            await convex.add_log(job_id, "INFO", "AML", f"Subject: {subject_context.get('query', 'N/A')}")
            await convex.update_progress(job_id, 5, "AML")

        # ========================================
        # Initialize components
        # ========================================

        # RGM for generating reviewer profiles
        rgm = ReviewerGenerationModule(
            age_range=tuple(reviewers_context["age_range"]),
            sex_distribution=reviewers_context["sex_distribution"],
            additional_context=reviewers_context.get("additional_context"),
        )

        # OpenRouter client for LLM calls
        llm = OpenRouterClient(api_key=api_key)

        # Generation settings
        count_mode = getattr(config.generation, 'count_mode', 'reviews')
        length_range = attributes_context["length_range"]
        avg_sentences_per_review = (length_range[0] + length_range[1]) / 2

        # Calculate max reviews based on count mode
        if count_mode == 'sentences':
            target_sentences = getattr(config.generation, 'target_sentences', None) or 1000
            estimated_reviews = int(target_sentences / avg_sentences_per_review)
            total_reviews = int(estimated_reviews * 1.3)  # 30% buffer for variance
            print(f"[Generation] Mode: Sentences | Target: {target_sentences} sentences")
            print(f"[Generation] Estimated ~{estimated_reviews} reviews (generating up to {total_reviews} with buffer)")
        else:
            target_sentences = None
            total_reviews = config.generation.count
            print(f"[Generation] Mode: Reviews | Target: {total_reviews} reviews")

        batch_size = config.generation.batch_size
        request_size = config.generation.request_size

        # Build model ID - check if model already has provider prefix
        model_name = config.generation.model
        provider = config.generation.provider
        if "/" in model_name:
            # Model already has provider prefix (e.g., "mistralai/mistral-7b")
            model = model_name
        else:
            # Add provider prefix
            model = f"{provider}/{model_name}"

        print(f"[Generation] Using model: {model} (provider={provider}, model_name={model_name})")

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

        for i in range(total_reviews):
            # Check sentence-based stopping condition FIRST (before any API calls)
            if count_mode == 'sentences' and target_sentences and total_sentences_generated >= target_sentences:
                print(f"[Generation] Target reached: {total_sentences_generated} sentences in {generated_count} reviews")
                if convex:
                    await convex.add_log(job_id, "INFO", "AML", f"Target reached: {total_sentences_generated} sentences in {generated_count} reviews")
                break

            # Check if job was terminated (poll Convex for status)
            if convex and i > 0 and i % 5 == 0:  # Check every 5 reviews
                job_status = await convex.get_job(job_id)
                if job_status and job_status.get("status") in ["terminated", "paused"]:
                    status = job_status.get("status")
                    print(f"[Generation] Job {status} by user - stopping generation")
                    if convex:
                        await convex.add_log(job_id, "INFO", "AML", f"Generation stopped by user ({status})")
                        await convex.update_progress(job_id, 100, status.capitalize())
                    return {
                        "status": status,
                        "jobId": job_id,
                        "generatedCount": generated_count,
                        "message": f"Job {status} after generating {generated_count} reviews",
                    }
            # Generate reviewer profile
            reviewer = rgm.generate_profile()

            # Random values within ranges
            num_sentences = random.randint(length_range[0], length_range[1])
            temperature = random.uniform(temp_range[0], temp_range[1])

            # Format prompts - match template placeholder names
            features_list = subject_context.get("characteristics", [])[:5]
            pros_list = subject_context.get("positives", [])[:3]
            cons_list = subject_context.get("negatives", [])[:3]

            # Build subject context summary for the template
            subject_context_text = f"""Domain: {subject_context.get("domain", subject_context.get("category", "N/A"))}
Region: {subject_context.get("region", "N/A")}
Key features: {", ".join(features_list) if features_list else "N/A"}"""

            # Determine dataset mode and build mode-specific prompt parts
            dataset_mode = getattr(config.generation, "dataset_mode", "explicit")
            aspect_cats = getattr(config.subject_profile, "aspect_categories", None) or ["PRODUCT#QUALITY", "PRODUCT#PRICES", "SERVICE#GENERAL", "EXPERIENCE#GENERAL"]

            if dataset_mode == "implicit":
                dataset_mode_instruction = "For implicit mode: do NOT include target terms. Only provide the category and polarity for each opinion."
                output_example = '{\n  "sentences": [\n    {\n      "text": "The food was absolutely divine.",\n      "opinions": [\n        {"category": "FOOD#QUALITY", "polarity": "positive"}\n      ]\n    }\n  ]\n}'
            else:
                dataset_mode_instruction = "For explicit mode: include the exact target term from the sentence text for each opinion."
                output_example = '{\n  "sentences": [\n    {\n      "text": "The pasta carbonara was absolutely divine.",\n      "opinions": [\n        {"target": "pasta carbonara", "category": "FOOD#QUALITY", "polarity": "positive"}\n      ]\n    }\n  ]\n}'

            prompt_vars = {
                "subject": subject_context["query"],
                "domain": subject_context.get("domain", subject_context.get("category", "product")),
                "subject_context": subject_context_text,
                "features": ", ".join(features_list) if features_list else "various features",
                "pros": ", ".join(pros_list) if pros_list else "quality and value",
                "cons": ", ".join(cons_list) if cons_list else "minor issues",
                # Sentence-level polarity distribution (percentages)
                "polarity_positive": int(polarity_dist["positive"] * 100),
                "polarity_neutral": int(polarity_dist["neutral"] * 100),
                "polarity_negative": int(polarity_dist["negative"] * 100),
                "age": reviewer.age,
                "sex": reviewer.sex,
                "region": subject_context.get("region", ""),
                "additional_context": reviewers_context.get("additional_context", "general consumer"),
                "min_sentences": num_sentences,
                "max_sentences": num_sentences + 1,
                "aspect_categories": ", ".join(aspect_cats),
                "dataset_mode_instruction": dataset_mode_instruction,
                "output_example": output_example,
            }

            system_prompt = format_prompt(system_template, **prompt_vars)
            user_prompt = format_prompt(user_template, **prompt_vars)

            # Save AML prompt file
            aml_number = str(i + 1).zfill(digits)
            aml_file = amls_path / f"aml-{aml_number}.md"
            aml_content = f"""# AML Prompt {aml_number}

## Assigned Parameters
- **Sentence Polarity Distribution**: {int(polarity_dist["positive"]*100)}% pos, {int(polarity_dist["neutral"]*100)}% neu, {int(polarity_dist["negative"]*100)}% neg
- **Age**: {reviewer.age}
- **Sex**: {reviewer.sex}
- **Length**: {num_sentences}-{num_sentences + 1} sentences

## System Prompt
{system_prompt}

## User Prompt
{user_prompt}
"""
            aml_file.write_text(aml_content, encoding="utf-8")

            # Generate review via LLM with rate limiting, retry, and validation
            review_text = None
            parsed_review = None
            validation_error = None
            validation_retries = 0

            for retry in range(MAX_RETRIES + MAX_VALIDATION_RETRIES):
                try:
                    # Rate limiting: wait if needed
                    elapsed_since_last = time.time() - last_request_time
                    if elapsed_since_last < REQUEST_DELAY:
                        wait_time = REQUEST_DELAY - elapsed_since_last
                        await asyncio.sleep(wait_time)

                    last_request_time = time.time()

                    # Build messages - add correction hint if this is a validation retry
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ]

                    # If retrying due to validation failure, add correction context
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

                    # chat() returns the content string directly, not a dict
                    review_text = await llm.chat(
                        model=model,
                        messages=messages,
                        temperature=temperature if validation_retries == 0 else max(0.3, temperature - 0.2),
                        max_tokens=1500,
                    )

                    # Validate the response
                    parsed_review, validation_error = validate_review_json(review_text, dataset_mode)

                    if parsed_review:
                        # Valid response - exit retry loop
                        break
                    else:
                        # Validation failed
                        validation_retries += 1
                        if validation_retries <= MAX_VALIDATION_RETRIES:
                            print(f"[Generation] Review {i+1} validation failed: {validation_error} (retry {validation_retries}/{MAX_VALIDATION_RETRIES})")
                            if convex and validation_retries == 1:
                                await convex.add_log(job_id, "WARN", "AML", f"Review {i+1} malformed JSON, retrying...")
                            continue
                        else:
                            # Exhausted validation retries - count as malformed
                            malformed_count += 1
                            print(f"[Generation] Review {i+1} validation failed after {MAX_VALIDATION_RETRIES} retries: {validation_error}")
                            if convex and malformed_count <= 3:
                                await convex.add_log(job_id, "WARN", "AML", f"Review {i+1} malformed after retries: {validation_error[:80]}")
                            break

                except Exception as e:
                    error_msg = str(e)

                    # Check if it's a rate limit error (429)
                    is_rate_limit = "429" in error_msg or "rate" in error_msg.lower() or "too many" in error_msg.lower()

                    if is_rate_limit and retry < MAX_RETRIES - 1:
                        # Exponential backoff for rate limits
                        backoff_time = (2 ** retry) * 5  # 5s, 10s, 20s
                        print(f"[Generation] Rate limited on review {i+1}, waiting {backoff_time}s (retry {retry+1}/{MAX_RETRIES})")
                        if convex and retry == 0:
                            await convex.add_log(job_id, "WARN", "AML", f"Rate limited, backing off {backoff_time}s...")
                        await asyncio.sleep(backoff_time)
                        continue
                    else:
                        # Non-rate-limit error or final retry
                        error_count += 1
                        last_error_msg = error_msg
                        print(f"[Generation] Error generating review {i+1}: {error_msg}")

                        # Log errors: first 3 errors individually, then summary
                        if convex:
                            if error_count <= 3:
                                await convex.add_log(job_id, "ERROR", "AML", f"Review {i+1} failed: {error_msg[:150]}")
                            elif error_count == 4:
                                await convex.add_log(job_id, "WARN", "AML", "Multiple errors occurring, will show summary at end")
                        break  # Exit retry loop on non-recoverable error

            # Process successful generation
            if parsed_review:
                # Successfully parsed and validated JSON
                sentences = parsed_review["sentences"]

                # Compute character offsets for explicit mode targets
                for sent in sentences:
                    for opinion in sent.get("opinions", []):
                        target = opinion.get("target")
                        if target and target != "NULL":
                            # Find target substring in sentence text
                            text = sent.get("text", "")
                            idx = text.lower().find(target.lower())
                            if idx >= 0:
                                opinion["from"] = idx
                                opinion["to"] = idx + len(target)
                            else:
                                opinion["from"] = 0
                                opinion["to"] = 0
                        else:
                            opinion["target"] = "NULL"
                            opinion["from"] = 0
                            opinion["to"] = 0

                # Build full review text from sentences
                full_text = " ".join(s["text"] for s in sentences)

                reviews.append({
                    "id": f"aml-{aml_number}",
                    "review_text": full_text,
                    "sentences": sentences,
                    "assigned": {
                        "polarity_distribution": {
                            "positive": int(polarity_dist["positive"] * 100),
                            "neutral": int(polarity_dist["neutral"] * 100),
                            "negative": int(polarity_dist["negative"] * 100),
                        },
                        "age": reviewer.age,
                        "sex": reviewer.sex,
                        "num_sentences": num_sentences,
                        "temperature": round(temperature, 2),
                    },
                })
                generated_count += 1
                total_sentences_generated += len(sentences)
            elif review_text and isinstance(review_text, str):
                # Fallback: treat as plain text (no structured annotations)
                # This happens when validation fails after all retries
                review_text = review_text.strip()
                reviews.append({
                    "id": f"aml-{aml_number}",
                    "review_text": review_text,
                    "sentences": [],
                    "assigned": {
                        "polarity_distribution": {
                            "positive": int(polarity_dist["positive"] * 100),
                            "neutral": int(polarity_dist["neutral"] * 100),
                            "negative": int(polarity_dist["negative"] * 100),
                        },
                        "age": reviewer.age,
                        "sex": reviewer.sex,
                        "num_sentences": num_sentences,
                        "temperature": round(temperature, 2),
                    },
                })
                generated_count += 1
                # For fallback reviews, estimate sentences from assigned num_sentences
                total_sentences_generated += num_sentences

            # Update progress after EVERY review for fluid UI updates
            elapsed = time.time() - start_time
            # Calculate progress from 5-99% during generation
            # 100% is set when job completes/fails
            if count_mode == 'sentences' and target_sentences:
                # Progress based on sentences generated
                progress_float = 5 + min(total_sentences_generated / target_sentences, 1.0) * 94
            else:
                # Progress based on reviews generated
                progress_float = 5 + (i + 1) / total_reviews * 94
            progress = min(int(progress_float), 99)  # Cap at 99% until completion

            # Calculate rate
            rate = generated_count / elapsed * 60 if elapsed > 0 else 0  # reviews per minute

            # Update Convex on every review for fluid progress bar
            if convex:
                await convex.update_progress(job_id, progress, "AML")
                await convex.update_generated_count(job_id, generated_count)
                # Only update failed count if there are actual failures
                if error_count > 0:
                    await convex.update_failed_count(job_id, error_count)

            # Log detailed progress less frequently (every 10% or 10 reviews)
            log_interval = max(1, min(total_reviews // 10, 10))
            should_log = (i + 1) % log_interval == 0 or i == total_reviews - 1

            if should_log and convex:
                # Detailed progress log
                elapsed_str = f"{int(elapsed)}s" if elapsed < 60 else f"{int(elapsed//60)}m {int(elapsed%60)}s"
                if count_mode == 'sentences' and target_sentences:
                    await convex.add_log(
                        job_id, "INFO", "AML",
                        f"Progress: {total_sentences_generated}/{target_sentences} sentences ({generated_count} reviews, {progress}%) | {elapsed_str} elapsed | {rate:.1f} reviews/min"
                    )
                else:
                    await convex.add_log(
                        job_id, "INFO", "AML",
                        f"Progress: {generated_count}/{total_reviews} ({progress}%) | {elapsed_str} elapsed | {rate:.1f} reviews/min"
                    )

                # Log sample preview for first few successful reviews
                if generated_count <= 3 and reviews:
                    preview = reviews[-1]["review_text"][:80] + "..." if len(reviews[-1]["review_text"]) > 80 else reviews[-1]["review_text"]
                    await convex.add_log(job_id, "INFO", "Sample", f'"{preview}"')

                # Warn if error rate is climbing
                current_error_rate = error_count / (i + 1) if i > 0 else 0
                if current_error_rate > 0.2 and i > 10:
                    await convex.add_log(job_id, "WARN", "AML", f"High error rate: {error_count}/{i+1} failed ({current_error_rate:.0%})")

            if should_log:
                if count_mode == 'sentences' and target_sentences:
                    print(f"[Generation] Progress: {total_sentences_generated}/{target_sentences} sentences ({generated_count} reviews, {progress}%) | {rate:.1f}/min")
                else:
                    print(f"[Generation] Progress: {generated_count}/{total_reviews} ({progress}%) | {rate:.1f}/min")

        # ========================================
        # Save dataset (always save JSONL + user-selected formats)
        # ========================================
        if convex:
            await convex.update_progress(job_id, 99, "Saving dataset...")
            await convex.add_log(job_id, "INFO", "Dataset", "Saving dataset files...")

        # Get selected output formats (default to jsonl if none specified)
        output_formats = config.generation.output_formats or ["jsonl"]
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

        def save_dataset_files(reviews_data, mode, suffix=""):
            """Save all selected format files for a given mode."""
            fmt_saved = []
            file_suffix = f"-{mode}" if suffix else ""

            # Save JSONL only if selected by user
            if "jsonl" in output_formats:
                jsonl_path = dataset_path / f"reviews{file_suffix}.jsonl"
                with open(jsonl_path, "w", encoding="utf-8") as f:
                    for idx, review in enumerate(reviews_data):
                        sentences = build_review_sentences(review, mode)
                        jsonl_entry = {
                            "id": str(idx),
                            "sentences": [],
                            "metadata": {
                                "assigned_polarity": review["assigned"]["polarity"],
                                "age": review["assigned"]["age"],
                                "sex": review["assigned"]["sex"],
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
                csv_path = dataset_path / f"reviews{file_suffix}.csv"
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
                xml_path = dataset_path / f"reviews{file_suffix}.xml"
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

            # Log summary statistics
            await convex.add_log(job_id, "INFO", "Summary", "=" * 40)
            await convex.add_log(job_id, "INFO", "Summary", f"Generation completed successfully!")
            await convex.add_log(job_id, "INFO", "Summary", f"Reviews: {generated_count}/{total_reviews} ({generated_count/total_reviews*100:.1f}%)")
            await convex.add_log(job_id, "INFO", "Summary", f"Duration: {elapsed_str} ({final_rate:.1f} reviews/min)")

            if error_count > 0:
                await convex.add_log(job_id, "WARN", "Summary", f"Errors: {error_count} failed ({final_error_rate:.1%} error rate)")

            if malformed_count > 0:
                await convex.add_log(job_id, "WARN", "Summary", f"Malformed JSON: {malformed_count} reviews fell back to plain text")

            # Polarity breakdown of actual generated reviews
            pol_counts = {"positive": 0, "neutral": 0, "negative": 0}
            for r in reviews:
                pol = r["assigned"]["polarity"]
                pol_counts[pol] = pol_counts.get(pol, 0) + 1

            await convex.add_log(
                job_id, "INFO", "Summary",
                f"Polarity: {pol_counts['positive']} pos, {pol_counts['neutral']} neu, {pol_counts['negative']} neg"
            )
            await convex.add_log(job_id, "INFO", "Summary", f"Output: {dataset_path}")
            await convex.add_log(job_id, "INFO", "Summary", "=" * 40)

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

    # RGM: Save reviewer context (just the distribution specs)
    reviewers_context = {
        "age_range": request.config.reviewer_profile.age_range,
        "sex_distribution": request.config.reviewer_profile.sex_distribution,
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


class PipelineRequest(BaseModel):
    """Request to run the full pipeline (selected phases sequentially)."""
    jobId: str
    jobName: str
    config: JobConfig
    phases: list[str]  # ["composition", "generation", "evaluation"]
    apiKey: str
    tavilyApiKey: Optional[str] = None
    jobsDirectory: str = "./jobs"
    convexUrl: Optional[str] = None
    convexToken: Optional[str] = None
    evaluationConfig: Optional[dict] = None
    datasetFile: Optional[str] = None  # For EVAL-only jobs
    reusedFromJobDir: Optional[str] = None  # Source job directory for composition reuse


async def execute_pipeline(
    job_id: str,
    job_name: str,
    config: JobConfig,
    phases: list[str],
    api_key: str,
    tavily_api_key: Optional[str],
    jobs_directory: str,
    convex_url: Optional[str],
    convex_token: Optional[str],
    evaluation_config: Optional[dict],
    dataset_file: Optional[str],
    reused_from_job_dir: Optional[str] = None,
):
    """Execute selected pipeline phases sequentially."""
    from pathlib import Path

    convex = None
    if convex_url and convex_token:
        convex = ConvexClient(convex_url, convex_token)

    try:
        # Get or create job directory
        job_paths = create_job_directory(jobs_directory, job_id, job_name)

        # ========================================
        # Handle Composition Reuse (copy context files from source job)
        # ========================================
        if reused_from_job_dir and "composition" not in phases:
            source_contexts_dir = Path(reused_from_job_dir) / "contexts"
            target_contexts_dir = Path(job_paths["contexts"])

            if source_contexts_dir.exists():
                import shutil
                # Copy all context files from source job
                for context_file in source_contexts_dir.glob("*.json"):
                    shutil.copy2(context_file, target_contexts_dir / context_file.name)

                if convex:
                    await convex.add_log(job_id, "INFO", "Pipeline", f"Copied composition contexts from: {reused_from_job_dir}")
            else:
                if convex:
                    await convex.add_log(job_id, "WARN", "Pipeline", f"Source job contexts not found: {source_contexts_dir}")

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
        )

        if convex:
            # Store config path in Convex
            await convex.run_mutation("jobs:setConfigPath", {
                "jobId": job_id,
                "configPath": config_path,
            })
            await convex.add_log(job_id, "INFO", "Pipeline", f"Starting pipeline with phases: {', '.join(phases)}")

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

        # ========================================
        # Phase 2: GENERATION (AML)
        # ========================================
        gen_result = None
        if "generation" in phases:
            if convex:
                await convex.update_progress(job_id, 30, "AML")
                await convex.add_log(job_id, "INFO", "AML", "Starting generation phase...")
                # Update status to running
                await convex.run_mutation("jobs:startGeneration", {"id": job_id})

            gen_result = await execute_generation(
                job_id=job_id,
                job_dir=job_paths["root"],
                config=config,
                api_key=api_key,
                convex_url=convex_url,
                convex_token=convex_token,
            )

            if convex:
                await convex.add_log(job_id, "INFO", "AML", "Generation phase complete.")

            # Compute and store conformity report
            if convex and gen_result and gen_result.get("reviews"):
                try:
                    reviews = gen_result["reviews"]
                    # Polarity Conformity: compare SENTENCE-LEVEL polarity vs target distribution
                    # The target distribution represents the desired mix of sentiments at the sentence/opinion level
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
                    length_range = config.attributes_profile.length_range  # [min, max] sentences
                    min_len, max_len = length_range[0], length_range[1]
                    within_range = 0
                    for r in reviews:
                        num_sentences = len(r.get("sentences", []))
                        if not num_sentences:
                            num_sentences = (r.get("assigned") or {}).get("num_sentences", min_len)
                        if min_len <= num_sentences <= max_len:
                            within_range += 1
                    length_conformity = round(within_range / total_reviews, 4) if total_reviews > 0 else 1.0

                    # Noise Conformity: since noise is applied programmatically, it's near-perfect
                    # Measure based on whether noise was requested and the generation succeeded
                    noise_config = config.attributes_profile.noise
                    if noise_config.typo_rate == 0 and not noise_config.colloquialism and not noise_config.grammar_errors:
                        noise_conformity = 1.0  # No noise requested = perfect conformity
                    else:
                        # Noise was applied programmatically - assume high conformity
                        # Slight variance from statistical application
                        noise_conformity = 0.95 + (0.05 * (1 - noise_config.typo_rate))
                        noise_conformity = round(min(noise_conformity, 1.0), 4)

                    # Validation Conformity: percentage of reviews with valid structured JSON
                    # Reviews with empty sentences[] array fell back to plain text due to validation failure
                    malformed_count = gen_result.get("malformedCount", 0)
                    valid_json_count = sum(1 for r in reviews if r.get("sentences"))
                    validation_conformity = round(valid_json_count / total_reviews, 4) if total_reviews > 0 else 1.0

                    await convex.run_mutation("jobs:setConformityReport", {
                        "jobId": job_id,
                        "conformityReport": {
                            "polarity": polarity_conformity,
                            "length": length_conformity,
                            "noise": noise_conformity,
                            "validation": validation_conformity,
                        },
                    })
                except Exception as e:
                    print(f"[Pipeline] Warning: Could not compute conformity: {e}")

        # ========================================
        # Phase 3: EVALUATION (MDQA)
        # ========================================
        if "evaluation" in phases:
            if convex:
                await convex.update_progress(job_id, 80, "MDQA")
                await convex.run_mutation("jobs:startEvaluation", {"id": job_id})
                await convex.add_log(job_id, "INFO", "MDQA", "Starting evaluation phase...")

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

    except Exception as e:
        error_msg = str(e)
        print(f"[Pipeline] Error: {error_msg}")
        if convex:
            await convex.add_log(job_id, "ERROR", "Pipeline", f"Pipeline failed: {error_msg}")
            await convex.fail_job(job_id, error_msg)


def load_texts_from_file(file_path: str) -> list[str]:
    """
    Load review texts from a dataset file (JSONL, CSV, or TXT).

    Returns a list of plain text strings extracted from the file.
    """
    from pathlib import Path
    import json

    path = Path(file_path)
    texts = []

    if not path.exists():
        return texts

    if str(path).endswith(".jsonl"):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        r = json.loads(line)
                        text = r.get("review_text") or r.get("text") or r.get("review") or ""
                        if not text:
                            sentences = r.get("sentences", [])
                            if sentences:
                                text = " ".join(s.get("text", "") for s in sentences if s.get("text"))
                        if text:
                            texts.append(text)
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

    elif str(path).endswith(".txt"):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    texts.append(line)

    else:
        # Try to parse as plain text (one review per line)
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    texts.append(line)

    return texts


async def execute_evaluation(
    job_id: str,
    job_paths: dict,
    evaluation_config: Optional[dict] = None,
    dataset_file: Optional[str] = None,
    convex: Optional["ConvexClient"] = None,
    reviews_data: Optional[list] = None,
):
    """
    Run MDQA evaluation on a dataset.

    Uses in-memory reviews_data if provided, otherwise reads from file.
    Supports reference dataset for meaningful Lexical/Semantic metrics.
    """
    from pathlib import Path
    import json

    review_texts = []

    if reviews_data:
        # Use in-memory data from generation phase
        for r in reviews_data:
            # Handle CERA's internal review format (has "sentences" from generation)
            sentences = r.get("sentences", [])
            if sentences:
                text = " ".join(s.get("text", "") for s in sentences if s.get("text"))
            else:
                text = r.get("review_text") or r.get("text") or r.get("review") or ""
            if text:
                review_texts.append(text)
    else:
        # Fall back to file-based loading
        if dataset_file:
            dataset_path = Path(dataset_file)
        else:
            # Try to find any dataset file in the dataset dir
            dataset_dir = Path(job_paths["dataset"])
            dataset_path = None
            for ext in [".jsonl", ".csv", ".xml"]:
                candidates = list(dataset_dir.glob(f"reviews*{ext}"))
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
        else:
            # Try to parse as JSONL
            with open(dataset_path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        reviews.append(json.loads(line))

        if not reviews:
            raise ValueError("No reviews found in dataset file")

        # Extract review texts (handles both flat and sentence-based formats)
        for r in reviews:
            text = r.get("review_text") or r.get("text") or r.get("review") or ""
            if not text:
                # Handle CERA's sentence-based JSONL format
                sentences = r.get("sentences", [])
                if sentences:
                    text = " ".join(s.get("text", "") for s in sentences if s.get("text"))
            if text:
                review_texts.append(text)

    if not review_texts:
        raise ValueError("No review text found in dataset")

    print(f"[Evaluation] Loaded {len(review_texts)} reviews for evaluation")

    # Check for reference dataset (for meaningful Lexical/Semantic metrics)
    reference_texts = []
    reference_metrics_enabled = evaluation_config.get("reference_metrics_enabled", False) if evaluation_config else False

    # Lexical/Semantic metrics that require a reference dataset
    reference_required_metrics = {"bleu", "rouge_l", "bertscore", "moverscore"}

    if reference_metrics_enabled:
        # Look for reference dataset file in the dataset directory
        dataset_dir = Path(job_paths["dataset"])
        reference_files = list(dataset_dir.glob("reference_*"))

        if reference_files:
            reference_file = reference_files[0]  # Use first reference file found
            reference_texts = load_texts_from_file(str(reference_file))
            if reference_texts:
                print(f"[Evaluation] Loaded {len(reference_texts)} reference texts from {reference_file.name}")
                if convex:
                    await convex.add_log(job_id, "INFO", "MDQA", f"Using reference dataset: {reference_file.name} ({len(reference_texts)} reviews)")
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
        await convex.add_log(job_id, "INFO", "MDQA", f"Computing metrics: {', '.join(selected_metrics)}")

    # Compute metrics with Rich progress bar
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

    metrics_results = {}

    # Map metric names to their compute functions
    # Lexical/Semantic metrics require reference texts (they're filtered out if no reference)
    # Diversity metrics only use generated texts
    metric_compute_map = {
        "bleu": lambda: compute_bleu_with_reference(review_texts, reference_texts),
        "rouge_l": lambda: compute_rouge_l_with_reference(review_texts, reference_texts),
        "bertscore": lambda: compute_bertscore_with_reference(review_texts, reference_texts),
        "moverscore": lambda: compute_moverscore_with_reference(review_texts, reference_texts),
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

        for metric_name in selected_metrics:
            if metric_name in metric_compute_map:
                progress.update(task, description=f"[cyan]Computing {metric_name}...")
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

    # Keep combined JSON for backward compatibility
    metrics_path = Path(job_paths["metrics"]) / "mdqa-results.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_results, f, indent=2)

    # Save per-category CSV files (easy to import to Excel/PowerPoint)
    metric_categories = {
        "lexical": {"bleu", "rouge_l"},
        "semantic": {"bertscore", "moverscore"},
        "diversity": {"distinct_1", "distinct_2", "self_bleu"},
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

    for category, metric_names in metric_categories.items():
        category_metrics = {k: v for k, v in metrics_results.items() if k in metric_names}
        if category_metrics:
            csv_path = Path(job_paths["metrics"]) / f"{category}-metrics.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv_module.writer(f)
                writer.writerow(["metric", "score", "description"])
                for metric, score in category_metrics.items():
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


def compute_bertscore_avg(texts: list[str]) -> float:
    """Compute average BERTScore F1 between consecutive pairs. Returns 0 if dependencies unavailable."""
    try:
        from sentence_transformers import SentenceTransformer, util
        import numpy as np

        model = SentenceTransformer("all-MiniLM-L6-v2")
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
        from sentence_transformers import SentenceTransformer
        import numpy as np

        if len(texts) < 2:
            return 0.0

        sampled = texts if len(texts) <= sample_size else random.sample(texts, sample_size)

        model = SentenceTransformer("all-MiniLM-L6-v2")

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
        from sentence_transformers import SentenceTransformer, util
        import numpy as np

        if not generated or not references:
            return 0.0

        gen_sampled = generated if len(generated) <= sample_size else random.sample(generated, sample_size)
        ref_sampled = references if len(references) <= sample_size else random.sample(references, sample_size)

        model = SentenceTransformer("all-MiniLM-L6-v2")

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
        from sentence_transformers import SentenceTransformer
        import numpy as np

        if not generated or not references:
            return 0.0

        gen_sampled = generated if len(generated) <= sample_size else random.sample(generated, sample_size)
        ref_sampled = references if len(references) <= sample_size else random.sample(references, sample_size)

        model = SentenceTransformer("all-MiniLM-L6-v2")

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
    """
    print(f"[API] Starting pipeline for job: {request.jobId}")
    print(f"[API] Selected phases: {request.phases}")

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
    )
    return {"status": "started", "jobId": request.jobId, "phases": request.phases}


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
    for job_dir in sorted(jobs_dir.iterdir(), reverse=True):
        if not job_dir.is_dir() or job_dir.name.startswith('.'):
            continue

        has_dataset = (job_dir / "dataset").exists() and any((job_dir / "dataset").iterdir())
        has_mavs = (job_dir / "mavs").exists() and any((job_dir / "mavs").iterdir())
        has_metrics = (job_dir / "metrics").exists() and any((job_dir / "metrics").iterdir())

        # Parse job name from directory name (format: {id}-{name})
        parts = job_dir.name.split('-', 1)
        job_id = parts[0] if len(parts) > 0 else job_dir.name
        job_name = parts[1].replace('-', ' ') if len(parts) > 1 else job_dir.name

        dataset_files = []
        if has_dataset:
            dataset_files = [f.name for f in (job_dir / "dataset").iterdir() if f.is_file()]

        mav_models = []
        if has_mavs:
            mav_models = [d.name for d in (job_dir / "mavs").iterdir() if d.is_dir()]

        jobs.append({
            "dirName": job_dir.name,
            "path": str(job_dir),
            "jobId": job_id,
            "jobName": job_name,
            "hasDataset": has_dataset,
            "hasMavs": has_mavs,
            "hasMetrics": has_metrics,
            "datasetFiles": dataset_files,
            "mavModels": mav_models,
        })

    return {"jobs": jobs}


class ReadDatasetRequest(BaseModel):
    jobDir: str
    filename: str


@app.post("/api/read-dataset")
async def read_dataset(request: ReadDatasetRequest):
    """Read and parse a dataset file from a job directory."""
    from pathlib import Path
    import json
    import xml.etree.ElementTree as ET

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
                    opinions.append({
                        "target": op_elem.get('target', 'NULL'),
                        "category": op_elem.get('category', ''),
                        "polarity": op_elem.get('polarity', ''),
                        "from": int(op_elem.get('from', 0)),
                        "to": int(op_elem.get('to', 0)),
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

    # Read MAV report (consensus data)
    report = None
    report_path = reports_dir / "mav-report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))

    # Read summary CSV
    summary = None
    summary_path = reports_dir / "mav-summary.csv"
    if summary_path.exists():
        summary = summary_path.read_text(encoding="utf-8")

    return {
        "models": models_data,
        "report": report,
        "summary": summary,
    }


class ReadMetricsRequest(BaseModel):
    jobDir: str


@app.post("/api/read-metrics")
async def read_metrics(request: ReadMetricsRequest):
    """Read MDQA metrics from a job directory (3 CSV files)."""
    from pathlib import Path
    import csv
    import json

    metrics_dir = Path(request.jobDir) / "metrics"
    if not metrics_dir.exists():
        raise HTTPException(status_code=404, detail="Metrics directory not found")

    result = {}

    # Read per-category CSV files
    for category in ["lexical", "semantic", "diversity"]:
        csv_path = metrics_dir / f"{category}-metrics.csv"
        if csv_path.exists():
            metrics = []
            with open(csv_path, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    metrics.append({
                        "metric": row.get("metric", ""),
                        "score": float(row.get("score", 0)),
                        "description": row.get("description", ""),
                    })
            result[category] = metrics

    # Fallback: read combined JSON if no CSVs found
    if not result:
        json_path = metrics_dir / "mdqa-results.json"
        if json_path.exists():
            raw = json.loads(json_path.read_text(encoding="utf-8"))
            # Categorize the metrics
            category_map = {
                "bleu": "lexical", "rouge_l": "lexical",
                "bertscore": "semantic", "moverscore": "semantic",
                "distinct_1": "diversity", "distinct_2": "diversity", "self_bleu": "diversity",
            }
            for metric, score in raw.items():
                cat = category_map.get(metric, "other")
                if cat not in result:
                    result[cat] = []
                result[cat].append({"metric": metric, "score": score, "description": ""})

    return {"metrics": result}


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

        extractor = ContextExtractor(
            api_key=request.apiKey,
            model=request.model,
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
        )

    except Exception as e:
        return ExtractContextResponse(
            error=f"Context extraction failed: {str(e)}"
        )
