"""CERA FastAPI Server - HTTP API for web GUI integration."""

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Union
import httpx
import asyncio
from datetime import datetime

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
    category: str
    feature_count: str
    sentiment_depth: str
    context_scope: str
    mav: Optional[MAVConfig] = None


class ReviewerProfile(BaseModel):
    age_range: list[int]
    sex_distribution: dict[str, float]
    audience_context: list[str]


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
    temp_range: list[float]


class GenerationConfig(BaseModel):
    count: int
    batch_size: int
    request_size: int = 5
    provider: str
    model: str


class JobConfig(BaseModel):
    subject_profile: SubjectProfile
    reviewer_profile: ReviewerProfile
    attributes_profile: AttributesProfile
    generation: GenerationConfig


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
    ├── composition/   # Composition context files
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
    subdirs = ["composition", "amls", "mavs", "metrics", "reports", "dataset"]
    paths = {"root": str(job_dir)}

    for subdir in subdirs:
        subdir_path = job_dir / subdir
        subdir_path.mkdir(parents=True, exist_ok=True)
        paths[subdir] = str(subdir_path)

    return paths


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
                category=config.subject_profile.category,
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
    - subject-context.json (from SIL)
    - reviewers-context.json (from RGM)
    - attributes-context.json (from ACM)

    Returns the job paths dictionary.
    """
    import json
    from pathlib import Path

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
        # Phase 1: SIL - Subject Intelligence Layer
        # ========================================
        if convex:
            await convex.update_composition_progress(job_id, 10, "SIL")
            await convex.add_log(
                job_id,
                "INFO",
                "SIL",
                f"Gathering intelligence for: {config.subject_profile.query}",
            )

        # TODO: Replace with actual SIL implementation
        # For now, create mock subject context
        subject_context = {
            "query": config.subject_profile.query,
            "region": config.subject_profile.region,
            "category": config.subject_profile.category,
            "feature_count": config.subject_profile.feature_count,
            "sentiment_depth": config.subject_profile.sentiment_depth,
            "context_scope": config.subject_profile.context_scope,
            "characteristics": [
                f"Feature 1 of {config.subject_profile.query}",
                f"Feature 2 of {config.subject_profile.query}",
                f"Feature 3 of {config.subject_profile.query}",
            ],
            "positives": [
                f"Positive aspect 1 of {config.subject_profile.query}",
                f"Positive aspect 2 of {config.subject_profile.query}",
            ],
            "negatives": [
                f"Negative aspect 1 of {config.subject_profile.query}",
            ],
            "use_cases": [
                f"Use case 1 for {config.subject_profile.query}",
                f"Use case 2 for {config.subject_profile.query}",
            ],
            "mav_enabled": config.subject_profile.mav.enabled if config.subject_profile.mav else False,
        }

        await asyncio.sleep(1)  # Simulated delay for SIL

        # Save subject context
        subject_context_path = Path(job_paths["composition"]) / "subject-context.json"
        with open(subject_context_path, "w") as f:
            json.dump(subject_context, f, indent=2)

        if convex:
            await convex.update_composition_progress(job_id, 35, "SIL")
            await convex.add_log(job_id, "INFO", "SIL", "Subject context saved")

        # ========================================
        # Phase 2: RGM - Reviewer Generation Module
        # ========================================
        if convex:
            await convex.update_composition_progress(job_id, 40, "RGM")
            await convex.add_log(
                job_id,
                "INFO",
                "RGM",
                f"Generating reviewer profiles (age: {config.reviewer_profile.age_range})",
            )

        # TODO: Replace with actual RGM implementation
        # For now, create mock reviewers context
        reviewers_context = {
            "age_range": config.reviewer_profile.age_range,
            "sex_distribution": config.reviewer_profile.sex_distribution,
            "audience_context": config.reviewer_profile.audience_context,
            "generated_count": config.generation.count,
            # Pre-generate reviewer assignments for each review
            "reviewer_assignments": [],
        }

        # Generate reviewer assignments based on distribution
        import random
        total_reviews = config.generation.count
        sex_dist = config.reviewer_profile.sex_distribution
        age_min, age_max = config.reviewer_profile.age_range

        for i in range(total_reviews):
            # Determine sex based on distribution
            rand = random.random()
            if rand < sex_dist.get("male", 0.5):
                sex = "male"
            elif rand < sex_dist.get("male", 0.5) + sex_dist.get("female", 0.5):
                sex = "female"
            else:
                sex = "unspecified"

            # Random age in range
            age = random.randint(age_min, age_max)

            reviewers_context["reviewer_assignments"].append({
                "id": i + 1,
                "age": age,
                "sex": sex,
                "context": random.choice(config.reviewer_profile.audience_context) if config.reviewer_profile.audience_context else None,
            })

        await asyncio.sleep(0.5)  # Simulated delay for RGM

        # Save reviewers context
        reviewers_context_path = Path(job_paths["composition"]) / "reviewers-context.json"
        with open(reviewers_context_path, "w") as f:
            json.dump(reviewers_context, f, indent=2)

        if convex:
            await convex.update_composition_progress(job_id, 65, "RGM")
            await convex.add_log(job_id, "INFO", "RGM", f"Generated {total_reviews} reviewer profiles")

        # ========================================
        # Phase 3: ACM - Attributes Composition Module
        # ========================================
        if convex:
            await convex.update_composition_progress(job_id, 70, "ACM")
            await convex.add_log(
                job_id,
                "INFO",
                "ACM",
                f"Composing attributes (polarity: {config.attributes_profile.polarity.positive:.0%} pos)",
            )

        # TODO: Replace with actual ACM implementation
        # For now, create mock attributes context
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
            "temp_range": config.attributes_profile.temp_range,
            # Pre-assign polarity for each review based on distribution
            "polarity_assignments": [],
        }

        # Generate polarity assignments based on distribution
        polarity_dist = config.attributes_profile.polarity
        for i in range(total_reviews):
            rand = random.random()
            if rand < polarity_dist.positive:
                polarity = "positive"
            elif rand < polarity_dist.positive + polarity_dist.neutral:
                polarity = "neutral"
            else:
                polarity = "negative"

            # Random length in range
            length_min, length_max = config.attributes_profile.length_range
            sentence_count = random.randint(length_min, length_max)

            # Random temperature in range
            temp_min, temp_max = config.attributes_profile.temp_range
            temperature = round(random.uniform(temp_min, temp_max), 2)

            attributes_context["polarity_assignments"].append({
                "id": i + 1,
                "polarity": polarity,
                "sentence_count": sentence_count,
                "temperature": temperature,
            })

        await asyncio.sleep(0.5)  # Simulated delay for ACM

        # Save attributes context
        attributes_context_path = Path(job_paths["composition"]) / "attributes-context.json"
        with open(attributes_context_path, "w") as f:
            json.dump(attributes_context, f, indent=2)

        if convex:
            await convex.update_composition_progress(job_id, 95, "ACM")
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


async def execute_composition_simple(
    job_id: str,
    job_name: str,
    config: JobConfig,
    api_key: str,
    tavily_api_key: Optional[str],
    jobs_directory: str,
) -> dict:
    """
    Execute composition phase WITHOUT calling Convex.
    Just creates files and returns results.
    The Convex action will handle all database updates.
    """
    import json
    from pathlib import Path
    import random

    # Create job directory structure
    job_paths = create_job_directory(jobs_directory, job_id, job_name)

    # ========================================
    # Phase 1: SIL - Subject Intelligence Layer
    # ========================================
    # TODO: Replace with actual SIL implementation
    subject_context = {
        "query": config.subject_profile.query,
        "region": config.subject_profile.region,
        "category": config.subject_profile.category,
        "feature_count": config.subject_profile.feature_count,
        "sentiment_depth": config.subject_profile.sentiment_depth,
        "context_scope": config.subject_profile.context_scope,
        "characteristics": [
            f"Feature 1 of {config.subject_profile.query}",
            f"Feature 2 of {config.subject_profile.query}",
            f"Feature 3 of {config.subject_profile.query}",
        ],
        "positives": [
            f"Positive aspect 1 of {config.subject_profile.query}",
            f"Positive aspect 2 of {config.subject_profile.query}",
        ],
        "negatives": [
            f"Negative aspect 1 of {config.subject_profile.query}",
        ],
        "use_cases": [
            f"Use case 1 for {config.subject_profile.query}",
            f"Use case 2 for {config.subject_profile.query}",
        ],
        "mav_enabled": config.subject_profile.mav.enabled if config.subject_profile.mav else False,
    }

    await asyncio.sleep(0.5)  # Simulated delay

    # Save subject context
    subject_context_path = Path(job_paths["composition"]) / "subject-context.json"
    with open(subject_context_path, "w") as f:
        json.dump(subject_context, f, indent=2)

    # ========================================
    # Phase 2: RGM - Reviewer Generation Module
    # ========================================
    total_reviews = config.generation.count
    sex_dist = config.reviewer_profile.sex_distribution
    age_min, age_max = config.reviewer_profile.age_range

    reviewers_context = {
        "age_range": config.reviewer_profile.age_range,
        "sex_distribution": config.reviewer_profile.sex_distribution,
        "audience_context": config.reviewer_profile.audience_context,
        "generated_count": total_reviews,
        "reviewer_assignments": [],
    }

    for i in range(total_reviews):
        rand = random.random()
        if rand < sex_dist.get("male", 0.5):
            sex = "male"
        elif rand < sex_dist.get("male", 0.5) + sex_dist.get("female", 0.5):
            sex = "female"
        else:
            sex = "unspecified"

        age = random.randint(age_min, age_max)
        reviewers_context["reviewer_assignments"].append({
            "id": i + 1,
            "age": age,
            "sex": sex,
            "context": random.choice(config.reviewer_profile.audience_context) if config.reviewer_profile.audience_context else None,
        })

    await asyncio.sleep(0.3)  # Simulated delay

    # Save reviewers context
    reviewers_context_path = Path(job_paths["composition"]) / "reviewers-context.json"
    with open(reviewers_context_path, "w") as f:
        json.dump(reviewers_context, f, indent=2)

    # ========================================
    # Phase 3: ACM - Attributes Composition Module
    # ========================================
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
        "temp_range": config.attributes_profile.temp_range,
        "polarity_assignments": [],
    }

    polarity_dist = config.attributes_profile.polarity
    for i in range(total_reviews):
        rand = random.random()
        if rand < polarity_dist.positive:
            polarity = "positive"
        elif rand < polarity_dist.positive + polarity_dist.neutral:
            polarity = "neutral"
        else:
            polarity = "negative"

        length_min, length_max = config.attributes_profile.length_range
        sentence_count = random.randint(length_min, length_max)

        temp_min, temp_max = config.attributes_profile.temp_range
        temperature = round(random.uniform(temp_min, temp_max), 2)

        attributes_context["polarity_assignments"].append({
            "id": i + 1,
            "polarity": polarity,
            "sentence_count": sentence_count,
            "temperature": temperature,
        })

    await asyncio.sleep(0.2)  # Simulated delay

    # Save attributes context
    attributes_context_path = Path(job_paths["composition"]) / "attributes-context.json"
    with open(attributes_context_path, "w") as f:
        json.dump(attributes_context, f, indent=2)

    return {
        "status": "composed",
        "jobId": job_id,
        "jobDir": job_paths["root"],
        "paths": job_paths,
    }


@app.post("/api/compose-job-simple")
async def compose_job_simple(request: CompositionRequestSimple):
    """
    Execute composition phase WITHOUT calling Convex.

    This endpoint is called by Convex actions which provide all needed data.
    Python only does file operations and returns results.
    The Convex action handles all database mutations.
    """
    try:
        result = await execute_composition_simple(
            request.jobId,
            request.jobName,
            request.config,
            request.apiKey,
            request.tavilyApiKey,
            request.jobsDirectory,
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
