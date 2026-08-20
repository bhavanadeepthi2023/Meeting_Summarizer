"""
Meeting Summarizer API.

Endpoints:
  POST   /api/meetings              - upload an audio file, starts async processing
  GET    /api/meetings/{job_id}     - poll job status (and full result once completed)
  GET    /api/meetings/{job_id}/transcript - raw transcript text (once available)
  DELETE /api/meetings/{job_id}     - discard a job record
  GET    /                          - static frontend (upload + polling UI)
"""
from __future__ import annotations

import logging
import uuid
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles

from app.audio_processing import AudioProcessingError, validate_extension
from app.config import get_settings
from app.job_manager import create_job, delete_job, get_job, purge_expired_jobs, run_pipeline
from app.models import JobResultResponse, JobStatus, JobSummaryResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

settings = get_settings()

app = FastAPI(
    title="Meeting Summarizer API",
    description="Transcribes meeting audio and extracts action-oriented summaries, decisions, and tasks.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_allow_origins),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/meetings", response_model=JobSummaryResponse, status_code=202)
async def upload_meeting(file: UploadFile, background_tasks: BackgroundTasks) -> JobSummaryResponse:
    if file.filename is None or file.filename.strip() == "":
        raise HTTPException(status_code=400, detail="Uploaded file has no filename.")

    try:
        validate_extension(file.filename)
    except AudioProcessingError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    job = create_job(original_filename=file.filename)
    work_dir = settings.upload_dir / job.job_id
    work_dir.mkdir(parents=True, exist_ok=True)
    raw_path = work_dir / f"raw_{uuid.uuid4().hex}{Path(file.filename).suffix.lower()}"

    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    bytes_written = 0
    try:
        with open(raw_path, "wb") as out_file:
            while chunk := await file.read(1024 * 1024):
                bytes_written += len(chunk)
                if bytes_written > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds the {settings.max_upload_size_mb}MB upload limit.",
                    )
                out_file.write(chunk)
    except HTTPException:
        raw_path.unlink(missing_ok=True)
        raise
    finally:
        await file.close()

    if bytes_written == 0:
        raw_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    background_tasks.add_task(run_pipeline, job.job_id, raw_path)

    return JobSummaryResponse(
        job_id=job.job_id,
        original_filename=job.original_filename,
        status=job.status,
        progress_message=job.progress_message,
        created_at=job.created_at,
        updated_at=job.updated_at,
        error=job.error,
        has_transcript=False,
        has_insights=False,
    )


@app.get("/api/meetings/{job_id}")
async def get_meeting(job_id: str) -> JobSummaryResponse | JobResultResponse:
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found. It may have expired or the ID is invalid.")

    if job.status == JobStatus.COMPLETED:
        if job.transcript is None or job.insights is None:
            # Should never happen, but guard against inconsistent state rather than 500ing opaquely.
            raise HTTPException(status_code=500, detail="Job marked completed but result data is missing.")
        return JobResultResponse(
            job_id=job.job_id,
            original_filename=job.original_filename,
            status=job.status,
            transcript=job.transcript,
            insights=job.insights,
        )

    return JobSummaryResponse(
        job_id=job.job_id,
        original_filename=job.original_filename,
        status=job.status,
        progress_message=job.progress_message,
        created_at=job.created_at,
        updated_at=job.updated_at,
        error=job.error,
        has_transcript=job.transcript is not None,
        has_insights=job.insights is not None,
    )


@app.get("/api/meetings/{job_id}/transcript", response_class=PlainTextResponse)
async def get_transcript(job_id: str) -> str:
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job.transcript is None:
        raise HTTPException(status_code=409, detail=f"Transcript not yet available (job status: {job.status.value}).")
    return job.transcript.full_text


@app.delete("/api/meetings/{job_id}", status_code=204)
async def delete_meeting(job_id: str) -> None:
    if not delete_job(job_id):
        raise HTTPException(status_code=404, detail="Job not found.")


@app.post("/api/admin/purge-expired")
async def purge_expired() -> dict[str, int]:
    removed = purge_expired_jobs()
    return {"removed": removed}


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


# Serve the lightweight frontend, if present, at the root path.
_frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if _frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(_frontend_dir), html=True), name="frontend")
