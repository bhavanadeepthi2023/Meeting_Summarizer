"""
LLM-based structured insight extraction.

Claude is called with a forced tool_choice pointing at a tool whose
input_schema is generated directly from the `MeetingInsights` Pydantic
model. This constrains generation to (near-)schema-conformant JSON. The
result is still independently validated against the Pydantic model — if
validation fails, we retry with the validation error fed back to the model
(bounded by `llm_extraction_max_retries`).

For transcripts too long to fit comfortably in a single extraction call, a
map-reduce strategy is used: the transcript is split into chunks, each is
summarized independently, and the summaries are combined into one final
structured extraction call.
"""
from __future__ import annotations

import logging

import anthropic
from anthropic import AsyncAnthropic
from pydantic import ValidationError

from app.config import get_settings
from app.models import MeetingInsights
from app.prompts import (
    CHUNK_SUMMARY_SYSTEM_PROMPT,
    CHUNK_SUMMARY_USER_TEMPLATE,
    EXTRACTION_SYSTEM_PROMPT,
    EXTRACTION_USER_TEMPLATE,
    REDUCE_USER_TEMPLATE,
)

logger = logging.getLogger(__name__)

_TOOL_NAME = "extract_meeting_insights"

RETRYABLE_ANTHROPIC_EXCEPTIONS = (
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
    anthropic.RateLimitError,
    anthropic.InternalServerError,
)


class InsightExtractionError(Exception):
    """Raised when structured insight extraction cannot be completed."""


def _build_tool_schema() -> dict:
    schema = MeetingInsights.model_json_schema()
    return {
        "name": _TOOL_NAME,
        "description": "Record the structured insights extracted from a meeting transcript.",
        "input_schema": schema,
    }


def _split_transcript(transcript: str, max_chars: int) -> list[str]:
    """Split on paragraph boundaries, keeping each chunk under max_chars where possible."""
    if len(transcript) <= max_chars:
        return [transcript]

    paragraphs = transcript.split("\n\n")
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        para_len = len(para) + 2
        if current and current_len + para_len > max_chars:
            chunks.append("\n\n".join(current))
            current, current_len = [], 0
        # A single paragraph longer than max_chars is hard-split to guarantee progress.
        if para_len > max_chars:
            for i in range(0, len(para), max_chars):
                chunks.append(para[i : i + max_chars])
            continue
        current.append(para)
        current_len += para_len

    if current:
        chunks.append("\n\n".join(current))

    return chunks


async def _call_with_retry(client: AsyncAnthropic, **kwargs) -> anthropic.types.Message:
    max_retries = 4
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            return await client.messages.create(**kwargs)
        except RETRYABLE_ANTHROPIC_EXCEPTIONS as exc:
            last_error = exc
            backoff = min(2 ** attempt, 30)
            logger.warning(
                "Claude call attempt %d/%d failed (%s); retrying in %ds",
                attempt, max_retries, exc.__class__.__name__, backoff,
            )
            import asyncio
            await asyncio.sleep(backoff)
        except anthropic.AuthenticationError as exc:
            raise InsightExtractionError("Anthropic authentication failed. Check ANTHROPIC_API_KEY.") from exc
        except anthropic.BadRequestError as exc:
            raise InsightExtractionError(f"Claude rejected the request: {exc}") from exc

    raise InsightExtractionError(f"Claude call failed after {max_retries} attempts: {last_error}") from last_error


async def _extract_structured(client: AsyncAnthropic, user_content: str) -> MeetingInsights:
    """Single forced-tool-use extraction call, with validation-retry loop."""
    settings = get_settings()
    tool = _build_tool_schema()

    messages = [{"role": "user", "content": user_content}]
    validation_feedback = ""

    for attempt in range(1, settings.llm_extraction_max_retries + 2):
        content = user_content if not validation_feedback else (
            f"{user_content}\n\n"
            f"Your previous response failed schema validation with this error:\n{validation_feedback}\n"
            f"Call the tool again with corrected input that satisfies the schema."
        )
        response = await _call_with_retry(
            client,
            model=settings.claude_model,
            max_tokens=settings.claude_max_tokens,
            system=EXTRACTION_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": content}],
            tools=[tool],
            tool_choice={"type": "tool", "name": _TOOL_NAME},
        )

        tool_use_block = next((b for b in response.content if b.type == "tool_use"), None)
        if tool_use_block is None:
            validation_feedback = "No tool_use block was returned."
            logger.warning("Claude did not return a tool_use block on attempt %d", attempt)
            continue

        try:
            return MeetingInsights.model_validate(tool_use_block.input)
        except ValidationError as exc:
            validation_feedback = str(exc)
            logger.warning("MeetingInsights validation failed on attempt %d: %s", attempt, exc)

    raise InsightExtractionError(
        f"Failed to obtain schema-valid insights after {settings.llm_extraction_max_retries + 1} attempts. "
        f"Last validation error: {validation_feedback}"
    )


async def _summarize_chunk(client: AsyncAnthropic, segment: str, index: int, total: int) -> str:
    settings = get_settings()
    response = await _call_with_retry(
        client,
        model=settings.claude_model,
        max_tokens=1500,
        system=CHUNK_SUMMARY_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": CHUNK_SUMMARY_USER_TEMPLATE.format(segment=segment, index=index, total=total),
            }
        ],
    )
    text_block = next((b for b in response.content if b.type == "text"), None)
    if text_block is None or not text_block.text.strip():
        raise InsightExtractionError(f"Claude returned no summary text for transcript segment {index}/{total}.")
    return text_block.text.strip()


async def extract_insights(transcript: str) -> MeetingInsights:
    if not transcript or not transcript.strip():
        raise InsightExtractionError("Cannot extract insights from an empty transcript.")

    settings = get_settings()
    client = AsyncAnthropic(api_key=settings.anthropic_api_key)

    try:
        chunks = _split_transcript(transcript, settings.transcript_chars_per_llm_call)

        if len(chunks) == 1:
            user_content = EXTRACTION_USER_TEMPLATE.format(transcript=chunks[0])
            return await _extract_structured(client, user_content)

        # Map-reduce path for long transcripts.
        logger.info("Transcript split into %d segments for map-reduce extraction.", len(chunks))
        summaries: list[str] = []
        for i, chunk in enumerate(chunks, start=1):
            summary = await _summarize_chunk(client, chunk, i, len(chunks))
            summaries.append(f"[Segment {i}/{len(chunks)}]\n{summary}")

        combined = "\n\n".join(summaries)
        reduce_content = REDUCE_USER_TEMPLATE.format(combined_summaries=combined)
        return await _extract_structured(client, reduce_content)
    finally:
        await client.close()
