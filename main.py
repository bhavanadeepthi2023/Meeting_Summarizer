import os
import shutil
import uuid
from typing import Any, Dict
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from ai_services import extract_structured_meeting_data, transcribe_audio_segments
from audio_processor import preprocess_and_split

app = FastAPI(title="Meeting Analysis Service", version="1.0.0")

JOB_STORE: Dict[str, Dict[str, Any]] = {}
STORAGE_ROOT = "./temp_storage"
os.makedirs(STORAGE_ROOT, exist_ok=True)


class JobSubmissionResponse(BaseModel):
    job_id: str
    status: str


def execute_pipeline(job_id: str, source_path: str) -> None:
    job_working_dir = os.path.join(STORAGE_ROOT, job_id)
    try:
        JOB_STORE[job_id]["status"] = "processing_audio"
        chunk_files = preprocess_and_split(source_path, job_working_dir)

        JOB_STORE[job_id]["status"] = "transcribing"
        transcript = transcribe_audio_segments(chunk_files)
        JOB_STORE[job_id]["transcript"] = transcript

        JOB_STORE[job_id]["status"] = "extracting_insights"
        structured_data = extract_structured_meeting_data(transcript)
        JOB_STORE[job_id]["result"] = structured_data.model_dump()
        JOB_STORE[job_id]["status"] = "completed"

    except Exception as exc:
        JOB_STORE[job_id]["status"] = "failed"
        JOB_STORE[job_id]["error"] = str(exc)
    finally:
        if os.path.exists(source_path):
            os.remove(source_path)
        if os.path.exists(job_working_dir):
            shutil.rmtree(job_working_dir, ignore_errors=True)


@app.post("/api/v1/jobs", response_model=JobSubmissionResponse)
async def submit_meeting_job(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    allowed_extensions = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".flac"}
    _, ext = os.path.splitext(file.filename or "")
    if ext.lower() not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{ext}'. Allowed: {', '.join(sorted(allowed_extensions))}",
        )

    job_id = str(uuid.uuid4())
    staged_path = os.path.join(STORAGE_ROOT, f"{job_id}_{file.filename}")

    with open(staged_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    JOB_STORE[job_id] = {
        "status": "queued",
        "transcript": None,
        "result": None,
        "error": None,
    }

    background_tasks.add_task(execute_pipeline, job_id, staged_path)
    return JobSubmissionResponse(job_id=job_id, status="queued")


@app.get("/api/v1/jobs/{job_id}")
async def get_job_state(job_id: str):
    if job_id not in JOB_STORE:
        raise HTTPException(status_code=404, detail="Job ID not found.")
    return JOB_STORE[job_id]