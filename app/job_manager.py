"""
In-memory job store and background pipeline orchestration.

This is deliberately simple: a thread-safe dict keyed by job_id, driven by
FastAPI's BackgroundTasks. It is appropriate for a single-process deployment.

To scale beyond one process/instance, replace `_JOBS` with a Redis-backed
store and replace the BackgroundTasks call in main.py with a Celery/RQ task
dispatch — `run_pipeline` below is already a self-contained unit of work and
can be lifted into a worker task with no other changes.
"""
from __future__ import annotations

import logging
import shutil
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

from app.audio_processing import AudioProcessingError, prepare_audio_chunks
from app.asr import TranscriptionError, transcribe_chunks
from app.config import get_settings
from app.llm_extraction import InsightExtractionError, extract_insights
from app.models import JobRecord, JobStatus

logger = logging.getLogger(__name__)

_JOBS: dict[str, JobRecord] = {}
_LOCK = threading.Lock()


def create_job(original_filename: str) -> JobRecord:
    job = JobRecord(original_filename=original_filename)
    with _LOCK:
        _JOBS[job.job_id] = job
    return job


def get_job(job_id: str) -> JobRecord | None:
    with _LOCK:
        job = _JOBS.get(job_id)
        return job.model_copy(deep=True) if job else None


def delete_job(job_id: str) -> bool:
    """Remove a job record and its working directory. Returns True if it existed."""
    with _LOCK:
        existed = _JOBS.pop(job_id, None) is not None
    if existed:
        _cleanup_work_dir(job_id)
    return existed


def _update(job_id: str, **fields) -> None:
    with _LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            return
        for key, value in fields.items():
            setattr(job, key, value)
        job.updated_at = datetime.now(timezone.utc)


def purge_expired_jobs() -> int:
    """Drop job records (and their working dirs) older than the retention window."""
    settings = get_settings()
    cutoff = datetime.now(timezone.utc) - timedelta(hours=settings.job_retention_hours)
    removed = 0
    with _LOCK:
        expired_ids = [jid for jid, job in _JOBS.items() if job.created_at < cutoff]
        for jid in expired_ids:
            del _JOBS[jid]
            removed += 1
    for jid in expired_ids:
        _cleanup_work_dir(jid)
    return removed


def _work_dir(job_id: str) -> Path:
    return get_settings().upload_dir / job_id


def _cleanup_work_dir(job_id: str) -> None:
    work_dir = _work_dir(job_id)
    if work_dir.exists():
        shutil.rmtree(work_dir, ignore_errors=True)


async def run_pipeline(job_id: str, raw_audio_path: Path) -> None:
    """
    The full ingest -> transcribe -> extract pipeline for one job.
    Every stage updates job status so clients can poll meaningful progress.
    Always cleans up the working directory, success or failure.
    """
    work_dir = _work_dir(job_id)
    try:
        _update(job_id, status=JobStatus.PREPROCESSING_AUDIO, progress_message="Normalizing and chunking audio")
        try:
            chunks = await prepare_audio_chunks(raw_audio_path, work_dir)
        except AudioProcessingError as exc:
            _update(job_id, status=JobStatus.FAILED, progress_message="Audio preprocessing failed", error=str(exc))
            return

        _update(
            job_id,
            status=JobStatus.TRANSCRIBING,
            progress_message=f"Transcribing {len(chunks)} audio chunk(s)",
        )
        try:
            transcript = await transcribe_chunks(chunks)
        except TranscriptionError as exc:
            _update(job_id, status=JobStatus.FAILED, progress_message="Transcription failed", error=str(exc))
            return

        _update(
            job_id,
            status=JobStatus.EXTRACTING_INSIGHTS,
            progress_message="Extracting summary, decisions, and action items",
            transcript=transcript,
        )
        try:
            insights = await extract_insights(transcript.full_text)
        except InsightExtractionError as exc:
            _update(
                job_id,
                status=JobStatus.FAILED,
                progress_message="Insight extraction failed",
                error=str(exc),
                transcript=transcript,
            )
            return

        _update(
            job_id,
            status=JobStatus.COMPLETED,
            progress_message="Done",
            insights=insights,
        )
    except Exception as exc:  # noqa: BLE001 - last-resort guard so a job never hangs in PROCESSING
        logger.exception("Unhandled error in pipeline for job %s", job_id)
        _update(job_id, status=JobStatus.FAILED, progress_message="Unexpected internal error", error=str(exc))
    finally:
        _cleanup_work_dir(job_id)
