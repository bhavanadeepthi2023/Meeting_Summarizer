# Meeting Summarizer

Transcribes meeting audio and extracts an executive summary, key decisions,
and a prioritized action-item list (assignee, deadline, priority) as
strictly validated structured data.

## Architecture

```
Upload (FastAPI) -> FFmpeg preprocessing (normalize + chunk)
                  -> OpenAI Whisper API (per-chunk transcription, stitched)
                  -> Claude API (forced tool-use, schema-validated extraction)
                  -> In-memory job store, polled by the frontend
```

- **ASR**: OpenAI Whisper API. Audio is normalized to mono 16kHz MP3 and
  split into fixed-duration chunks (default 10 min) via FFmpeg, since
  Whisper caps uploads at 25MB. Chunks are transcribed sequentially with
  retry/backoff on transient errors, then stitched into one transcript.
- **Insight extraction**: Claude, called with a forced `tool_choice` whose
  `input_schema` is generated directly from the `MeetingInsights` Pydantic
  model (`app/models.py`). The tool-call output is independently re-validated
  against that same model; on validation failure, the error is fed back to
  Claude and the call is retried (bounded by `LLM_EXTRACTION_MAX_RETRIES`).
- **Long transcripts**: if a transcript exceeds `TRANSCRIPT_CHARS_PER_LLM_CALL`
  (default ~60k chars), it's split on paragraph boundaries, each segment is
  summarized independently, and the summaries are combined in one final
  structured extraction call (map-reduce), so meeting length isn't bounded
  by a single context window.
- **Job orchestration**: FastAPI `BackgroundTasks` + a thread-safe in-memory
  dict (`app/job_manager.py`). This is intentionally simple for a
  single-process deployment. `run_pipeline()` is a self-contained async
  function — to scale to multiple workers/instances, swap the dict for a
  Redis-backed store and dispatch `run_pipeline` as a Celery/RQ task instead
  of a `BackgroundTasks` call; no other code changes needed.

## Setup

```bash
cp .env.example .env
# edit .env and set OPENAI_API_KEY and ANTHROPIC_API_KEY

pip install -r requirements.txt
# ffmpeg must be installed and on PATH (apt-get install ffmpeg / brew install ffmpeg)

uvicorn app.main:app --reload
```

Open `http://localhost:8000` for the upload UI, or `http://localhost:8000/docs`
for the interactive API docs.

### Docker

```bash
docker build -t meeting-summarizer .
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=sk-... \
  -e ANTHROPIC_API_KEY=sk-ant-... \
  meeting-summarizer
```

## API

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/meetings` | Upload an audio file (multipart `file` field). Returns `202` with a `job_id` immediately; processing runs in the background. |
| `GET` | `/api/meetings/{job_id}` | Poll status. While in progress, returns a status summary; once `status: "completed"`, returns the full transcript + insights. |
| `GET` | `/api/meetings/{job_id}/transcript` | Plain-text transcript (once available). |
| `DELETE` | `/api/meetings/{job_id}` | Discard a job record and its temp files. |
| `GET` | `/api/health` | Liveness check. |

Example:

```bash
curl -X POST http://localhost:8000/api/meetings -F "file=@standup.m4a"
# {"job_id": "...", "status": "pending", ...}

curl http://localhost:8000/api/meetings/<job_id>
# poll until status == "completed"
```

### Response shape once completed

```json
{
  "job_id": "...",
  "status": "completed",
  "transcript": { "full_text": "...", "segments": [...], "duration_seconds": 1834.2 },
  "insights": {
    "executive_summary": "...",
    "key_decisions": [{ "decision": "...", "context": "..." }],
    "action_items": [
      { "task": "...", "assignee": "Priya", "deadline": "2026-08-27", "priority": "high" }
    ],
    "attendees_mentioned": ["..."],
    "open_questions": ["..."]
  }
}
```

## Error handling

- **Unsupported/corrupt audio**: rejected at upload (bad extension) or during
  ffprobe validation (unreadable/no audio stream) — job fails fast with a
  specific message, never a silent hang.
- **Oversized uploads**: rejected at `MAX_UPLOAD_SIZE_MB` (default 500MB)
  before any processing starts.
- **Whisper/Claude transient failures**: retried with exponential backoff;
  authentication and bad-request errors fail immediately with a clear cause
  rather than retrying pointlessly.
- **LLM schema drift**: output is validated against `MeetingInsights`; on
  failure, the validation error is sent back to Claude for a corrected
  retry (bounded, then fails the job rather than returning malformed data).
- **Any unhandled exception** in the pipeline is caught at the top level so
  a job always resolves to `completed` or `failed` — it can't hang
  indefinitely — and the working directory is always cleaned up.

## Configuration

All settings are environment variables (see `.env.example`), validated at
startup via `app/config.py`. Notably:

- `WHISPER_CHUNK_SECONDS` — audio chunk duration sent per Whisper call.
- `TRANSCRIPT_CHARS_PER_LLM_CALL` — threshold before map-reduce summarization kicks in.
- `JOB_RETENTION_HOURS` — how long completed job records stay queryable in memory.

## Known limitations / production hardening notes

- The in-memory job store does not survive a process restart and does not
  work across multiple uvicorn workers — see the scaling note above.
- No authentication is implemented on the API — add an auth layer
  (API key, OAuth, etc.) before exposing this publicly.
- Whisper does not return speaker diarization; action-item assignees are
  inferred by Claude from names mentioned in speech, not from speaker
  labels. For diarized transcripts, swap in Azure Speech or a diarization-
  capable ASR provider.
