"""
Speech-to-text via the OpenAI Whisper API.

Each audio chunk is transcribed independently (with retry/backoff on
transient errors) and the results are stitched into a single transcript,
preserving per-chunk offsets so downstream consumers can still reason about
timing if needed.
"""
from __future__ import annotations

import asyncio
import logging

import openai
from openai import AsyncOpenAI

from app.audio_processing import AudioChunk
from app.config import get_settings
from app.models import TranscriptionResult, TranscriptSegment

logger = logging.getLogger(__name__)


class TranscriptionError(Exception):
    """Raised when transcription cannot be completed after retries."""


RETRYABLE_EXCEPTIONS = (
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.RateLimitError,
    openai.InternalServerError,
)


async def _transcribe_chunk_with_retry(
    client: AsyncOpenAI, chunk: AudioChunk, model: str, max_retries: int = 4
) -> str:
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            with open(chunk.path, "rb") as audio_file:
                response = await client.audio.transcriptions.create(
                    model=model,
                    file=audio_file,
                    response_format="text",
                )
            # SDK returns a plain string when response_format="text"
            text = response if isinstance(response, str) else getattr(response, "text", str(response))
            return text.strip()
        except RETRYABLE_EXCEPTIONS as exc:
            last_error = exc
            backoff = min(2 ** attempt, 30)
            logger.warning(
                "Whisper transcription attempt %d/%d failed for chunk %d (%s); retrying in %ds",
                attempt, max_retries, chunk.chunk_index, exc.__class__.__name__, backoff,
            )
            await asyncio.sleep(backoff)
        except openai.AuthenticationError as exc:
            raise TranscriptionError(
                "OpenAI authentication failed. Check OPENAI_API_KEY."
            ) from exc
        except openai.BadRequestError as exc:
            raise TranscriptionError(
                f"Whisper rejected chunk {chunk.chunk_index}: {exc}"
            ) from exc

    raise TranscriptionError(
        f"Chunk {chunk.chunk_index} failed after {max_retries} attempts: {last_error}"
    ) from last_error


async def transcribe_chunks(chunks: list[AudioChunk]) -> TranscriptionResult:
    if not chunks:
        raise TranscriptionError("No audio chunks provided for transcription.")

    settings = get_settings()
    client = AsyncOpenAI(api_key=settings.openai_api_key)

    # Sequential, not parallel: keeps us well under Whisper API rate limits
    # for long meetings with many chunks, and keeps error attribution simple.
    segments: list[TranscriptSegment] = []
    try:
        for chunk in chunks:
            text = await _transcribe_chunk_with_retry(client, chunk, settings.whisper_model)
            segments.append(
                TranscriptSegment(
                    chunk_index=chunk.chunk_index,
                    start_offset_seconds=chunk.start_offset_seconds,
                    end_offset_seconds=chunk.end_offset_seconds,
                    text=text,
                )
            )
    finally:
        await client.close()

    full_text = "\n\n".join(s.text for s in segments if s.text)
    if not full_text.strip():
        raise TranscriptionError("Transcription completed but produced no text (silent or unintelligible audio).")

    total_duration = segments[-1].end_offset_seconds if segments else 0.0

    return TranscriptionResult(
        full_text=full_text,
        segments=segments,
        duration_seconds=total_duration,
        detected_language=None,
    )
