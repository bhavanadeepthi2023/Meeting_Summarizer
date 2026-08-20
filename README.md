# Meeting Summarizer

Transcribes meeting audio locally and extracts an executive summary, key decisions, discussion topics, and a prioritized action-item list (assignee, deadline, priority) as strictly validated structured data.

## Architecture

```
Upload (Streamlit / FastAPI) -> FFmpeg preprocessing (16kHz mono normalize + chunking)
                              -> Faster-Whisper (local int8 transcription, stitched)
                              -> Groq API (Llama 3 with JSON schema-validated extraction)
                              -> In-memory job store, polled by the frontend
```

- **ASR**: Local `faster-whisper` (`base` model, int8 CPU/GPU quantization). Audio is normalized to mono 16kHz 64kbps MP3 and partitioned into sub-24MB chunks via FFmpeg. Chunks are transcribed locally and stitched into a unified transcript with zero external speech API billing.
- **Insight extraction**: Groq API (`llama-3.3-70b-versatile` / `llama-3.1-8b-instant`), called with JSON mode. The structured output is validated directly against the `MeetingSummaryResult` Pydantic model (`schemas.py`).
- **Job orchestration**: FastAPI `BackgroundTasks` + a thread-safe in-memory dictionary (`JOB_STORE` in `main.py`). `execute_pipeline()` runs asynchronously in the background. To scale across multiple worker processes, replace the in-memory dictionary with Redis and dispatch tasks via Celery.
- **Frontend**: Streamlit UI (`app.py`) providing file drag-and-drop, state polling (`processing_audio` -> `transcribing` -> `extracting_insights` -> `completed`), and structured data presentation.

## Setup

```bash
# Clone and enter directory
git clone https://github.com/<your-username>/meeting-summarizer.git
cd meeting-summarizer

# Create and activate virtual environment
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate

# Install dependencies (requires FFmpeg on system PATH)
pip install -r requirements.txt

# Set Groq API Key
# Windows: $env:GROQ_API_KEY="gsk_..."
# macOS/Linux: export GROQ_API_KEY="gsk_..."
```

## Running the Services

Start the backend API and frontend UI in two separate terminals:

```bash
# Terminal 1: Backend API
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload

# Terminal 2: Streamlit Dashboard
python -m streamlit run app.py
```

Open `http://localhost:8501` for the upload UI, or `http://127.0.0.1:8000/docs` for the interactive OpenAPI documentation.

## API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/v1/jobs` | Upload an audio file (multipart `file` field). Returns `200` with a `job_id` immediately; processing runs in the background. |
| `GET` | `/api/v1/jobs/{job_id}` | Poll status. Returns active pipeline status, and once `status: "completed"`, returns the full transcript + structured insights. |

Example:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/jobs -F "file=@meeting_recording.mp3"
# {"job_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479", "status": "queued"}

curl http://127.0.0.1:8000/api/v1/jobs/f47ac10b-58cc-4372-a567-0e02b2c3d479
# poll until status == "completed"
```

### Response shape once completed

```json
{
  "status": "completed",
  "transcript": "Alex: Let's finalize the database strategy...",
  "result": {
    "meeting_title": "Sprint 14 Planning & Architecture Sync",
    "executive_summary": "The team agreed to transition connection pooling to PgBouncer to resolve thread saturation during peak batch loads.",
    "key_topics": [
      "Database connection pooling",
      "API rate limiting configuration",
      "Staging release timeline"
    ],
    "decisions": [
      {
        "decision": "Deploy PgBouncer for database connection pooling",
        "rationale": "Mitigates thread saturation during high-concurrency operations"
      }
    ],
    "action_items": [
      {
        "task": "Configure PgBouncer helm charts on staging",
        "assignee": "Alex",
        "deadline": "Friday 17:00",
        "priority": "high"
      }
    ]
  },
  "error": null
}
```

## Error handling

- **Unsupported/corrupt audio**: Rejected immediately on upload validation (`400 Bad Request`) if the file extension does not match allowed audio containers (`.mp3`, `.wav`, `.m4a`, `.aac`, `.ogg`, `.flac`).
- **Model routing fallbacks**: `resolve_active_model()` dynamically queries Groq's active model registry to prevent execution failure caused by deprecated model identifiers.
- **LLM schema validation**: Raw JSON responses are validated through Pydantic (`MeetingSummaryResult.model_validate`). Parsing errors raise structured validation exceptions rather than propagating silent data corruption.
- **Lifecycle cleanup**: Temporary normalized chunk directories and uploaded files are purged inside a mandatory `finally` block regardless of pipeline success or failure.

## Configuration

All credentials and pipeline settings are configured via environment variables:

- `GROQ_API_KEY` — API key from Groq Console used for structured insight extraction.
- `MAX_CHUNK_BYTES` — Maximum size threshold per audio segment before slicing (default 24MB).

## Known limitations / production hardening notes

- The in-memory `JOB_STORE` dictionary does not survive a server restart and cannot be shared across multiple Uvicorn workers. Replace with a Redis instance for production multi-worker scaling.
- Faster-Whisper does not perform speaker diarization; action-item assignees are extracted via LLM contextual inference from names referenced in dialogue.
- Authentication is not enabled by default on the API routes; add JWT/OAuth2 middleware before exposing the service on a public network.
