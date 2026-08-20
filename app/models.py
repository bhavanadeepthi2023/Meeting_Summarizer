"""
Pydantic data models shared across the pipeline.

`MeetingInsights` is the strict contract the LLM extraction step must satisfy.
It is used both to validate Claude's tool-call output and to generate the
JSON schema handed to Claude as a tool definition, so the model is
constrained at generation time as well as validated after the fact.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field, field_validator


# --------------------------------------------------------------------------- #
# Job lifecycle
# --------------------------------------------------------------------------- #
class JobStatus(str, Enum):
    PENDING = "pending"
    VALIDATING = "validating"
    PREPROCESSING_AUDIO = "preprocessing_audio"
    TRANSCRIBING = "transcribing"
    EXTRACTING_INSIGHTS = "extracting_insights"
    COMPLETED = "completed"
    FAILED = "failed"


# --------------------------------------------------------------------------- #
# Transcription
# --------------------------------------------------------------------------- #
class TranscriptSegment(BaseModel):
    chunk_index: int
    start_offset_seconds: float
    end_offset_seconds: float
    text: str


class TranscriptionResult(BaseModel):
    full_text: str
    segments: list[TranscriptSegment]
    duration_seconds: float
    detected_language: str | None = None


# --------------------------------------------------------------------------- #
# Structured meeting insights (strict LLM output contract)
# --------------------------------------------------------------------------- #
class Priority(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Decision(BaseModel):
    decision: str = Field(..., min_length=1, description="The decision that was made, stated concretely")
    context: str | None = Field(default=None, description="Brief context or rationale behind the decision, if discussed")


class ActionItem(BaseModel):
    task: str = Field(..., min_length=1, description="A single, concrete, actionable task")
    assignee: str | None = Field(default=None, description="Person or team responsible; null if not stated in the meeting")
    deadline: str | None = Field(
        default=None,
        description="Deadline as stated in the meeting (ISO 8601 date if determinable, otherwise the relative phrase used, e.g. 'next Friday'); null if not mentioned",
    )
    priority: Priority = Field(..., description="Inferred urgency based on language and context")

    @field_validator("task", "assignee", "deadline")
    @classmethod
    def _strip(cls, v: str | None) -> str | None:
        if v is None:
            return v
        v = v.strip()
        return v or None


class MeetingInsights(BaseModel):
    executive_summary: str = Field(..., min_length=1, description="3-6 sentence executive summary of the meeting")
    key_decisions: list[Decision] = Field(default_factory=list)
    action_items: list[ActionItem] = Field(default_factory=list)
    attendees_mentioned: list[str] = Field(
        default_factory=list, description="Names of participants explicitly identifiable from the transcript"
    )
    open_questions: list[str] = Field(
        default_factory=list, description="Unresolved questions or topics explicitly flagged as needing follow-up"
    )


# --------------------------------------------------------------------------- #
# Job record (full pipeline state)
# --------------------------------------------------------------------------- #
class JobRecord(BaseModel):
    job_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    original_filename: str
    status: JobStatus = JobStatus.PENDING
    progress_message: str = "Queued"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    error: str | None = None

    transcript: TranscriptionResult | None = None
    insights: MeetingInsights | None = None

    class Config:
        use_enum_values = False


class JobSummaryResponse(BaseModel):
    """Slim response returned by the status-polling endpoint."""
    job_id: str
    original_filename: str
    status: JobStatus
    progress_message: str
    created_at: datetime
    updated_at: datetime
    error: str | None = None
    has_transcript: bool
    has_insights: bool


class JobResultResponse(BaseModel):
    """Full response returned once a job completes."""
    job_id: str
    original_filename: str
    status: JobStatus
    transcript: TranscriptionResult
    insights: MeetingInsights
