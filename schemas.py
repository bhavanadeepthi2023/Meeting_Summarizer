from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class PriorityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ActionItem(BaseModel):
    task: str = Field(..., description="Actionable statement describing the assigned task.")
    assignee: Optional[str] = Field(
        default=None,
        description="Full name or team identifier of the individual responsible for delivery.",
    )
    deadline: Optional[str] = Field(
        default=None,
        description="Explicit or contextually inferred deadline or timeline.",
    )
    priority: PriorityLevel = Field(
        default=PriorityLevel.MEDIUM,
        description="Assigned task priority.",
    )


class KeyDecision(BaseModel):
    decision: str = Field(..., description="The definitive agreement or decision reached.")
    rationale: Optional[str] = Field(
        default=None,
        description="Underlying business or technical justification.",
    )


class MeetingSummaryResult(BaseModel):
    meeting_title: str = Field(..., description="Concise, descriptive title for the meeting session.")
    executive_summary: str = Field(
        ...,
        description="Concise summary covering meeting objectives, main debate points, and final consensus.",
    )
    key_topics: List[str] = Field(
        ...,
        description="List of primary topics addressed during the session.",
    )
    decisions: List[KeyDecision] = Field(
        default_factory=list,
        description="Collection of formalized decisions.",
    )
    action_items: List[ActionItem] = Field(
        default_factory=list,
        description="Structured tasks and deliverables extracted from the dialogue.",
    )