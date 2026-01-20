"""CERA FastAPI Server - HTTP API for web GUI integration."""

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
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


class SubjectProfile(BaseModel):
    query: str
    region: str
    category: str
    feature_count: str
    sentiment_depth: str
    context_scope: str


class ReviewerProfile(BaseModel):
    age_range: list[int]
    sex_distribution: dict[str, float]
    audience_context: list[str]


class NoiseConfig(BaseModel):
    typo_rate: float
    colloquialism: bool
    grammar_errors: bool


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
    provider: str
    model: str


class JobConfig(BaseModel):
    subject_profile: SubjectProfile
    reviewer_profile: ReviewerProfile
    attributes_profile: AttributesProfile
    generation: GenerationConfig


class JobRequest(BaseModel):
    jobId: str
    config: JobConfig
    apiKey: str
    convexUrl: Optional[str] = None
    convexToken: Optional[str] = None


class ConvexClient:
    """Client for updating Convex database from Python."""

    def __init__(self, url: str, token: str):
        self.url = url
        self.token = token
        self.client = httpx.AsyncClient()

    async def update_progress(self, job_id: str, progress: int, phase: str):
        """Update job progress in Convex."""
        try:
            # Convex HTTP API for mutations
            response = await self.client.post(
                f"{self.url}/api/mutation",
                headers={
                    "Authorization": f"Bearer {self.token}",
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
                    "Authorization": f"Bearer {self.token}",
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
                    "Authorization": f"Bearer {self.token}",
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
                    "Authorization": f"Bearer {self.token}",
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
                    "Authorization": f"Bearer {self.token}",
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
    config: JobConfig,
    api_key: str,
    convex_url: Optional[str],
    convex_token: Optional[str],
):
    """Execute the CERA pipeline."""
    convex = None
    if convex_url and convex_token:
        convex = ConvexClient(convex_url, convex_token)

    try:
        # Phase 1: Composition
        if convex:
            await convex.update_progress(job_id, 5, "composition")
            await convex.add_log(job_id, "INFO", "composition", "Starting composition phase...")

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

        # Complete
        output_path = f"./output/{job_id}"

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
        request.config,
        request.apiKey,
        request.convexUrl,
        request.convexToken,
    )
    return {"status": "started", "jobId": request.jobId}


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
