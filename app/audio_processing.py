"""
Audio preprocessing: format validation, normalization, and chunking.

Whisper's API caps uploads at 25MB. Meeting recordings routinely exceed that,
so every file is normalized to a mono 16kHz MP3 (small, ASR-friendly) and
then split into fixed-duration chunks that are each guaranteed to sit under
the size limit. Chunk boundaries carry their offset so transcripts can be
stitched back together with correct timestamps.
"""
from __future__ import annotations

import asyncio
import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from app.config import get_settings

logger = logging.getLogger(__name__)


class AudioProcessingError(Exception):
    """Raised for any unrecoverable audio validation/processing failure."""


@dataclass
class AudioChunk:
    path: Path
    chunk_index: int
    start_offset_seconds: float
    end_offset_seconds: float


def _require_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise AudioProcessingError(
            "ffmpeg/ffprobe not found on PATH. Install ffmpeg (e.g. `apt-get install ffmpeg`) "
            "in the runtime environment."
        )


def validate_extension(filename: str) -> None:
    settings = get_settings()
    suffix = Path(filename).suffix.lower()
    if suffix not in settings.allowed_audio_extensions:
        raise AudioProcessingError(
            f"Unsupported file extension '{suffix}'. Allowed: {', '.join(settings.allowed_audio_extensions)}"
        )


def probe_duration_seconds(file_path: Path) -> float:
    """Run ffprobe to get media duration; also acts as a corruption/format check."""
    _require_ffmpeg()
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-print_format", "json",
                "-show_format", "-show_streams", str(file_path),
            ],
            capture_output=True, text=True, timeout=60, check=True,
        )
    except subprocess.TimeoutExpired as exc:
        raise AudioProcessingError("ffprobe timed out inspecting the file; it may be corrupt or truncated.") from exc
    except subprocess.CalledProcessError as exc:
        raise AudioProcessingError(
            f"File could not be read as a valid audio/video container. ffprobe error: {exc.stderr.strip()}"
        ) from exc

    try:
        info = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise AudioProcessingError("ffprobe returned unparseable output.") from exc

    has_audio_stream = any(s.get("codec_type") == "audio" for s in info.get("streams", []))
    if not has_audio_stream:
        raise AudioProcessingError("No audio stream detected in the uploaded file.")

    duration_str = info.get("format", {}).get("duration")
    if duration_str is None:
        raise AudioProcessingError("Could not determine audio duration from file metadata.")

    try:
        return float(duration_str)
    except ValueError as exc:
        raise AudioProcessingError(f"Invalid duration value reported by ffprobe: {duration_str!r}") from exc


def normalize_to_mono_mp3(src_path: Path, dst_path: Path, bitrate_kbps: int = 64) -> None:
    """
    Downmix to mono, resample to 16kHz, and encode as MP3 at a low bitrate.
    This is sufficient quality for ASR and keeps chunk sizes small and predictable.
    """
    _require_ffmpeg()
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(src_path),
                "-ac", "1", "-ar", "16000",
                "-b:a", f"{bitrate_kbps}k",
                "-vn",
                str(dst_path),
            ],
            capture_output=True, text=True, timeout=1800, check=True,
        )
    except subprocess.TimeoutExpired as exc:
        raise AudioProcessingError("Audio normalization timed out (file too large or ffmpeg stalled).") from exc
    except subprocess.CalledProcessError as exc:
        raise AudioProcessingError(f"ffmpeg failed to normalize audio: {exc.stderr.strip()}") from exc

    if not dst_path.exists() or dst_path.stat().st_size == 0:
        raise AudioProcessingError("Normalization produced an empty output file.")


def split_into_chunks(normalized_path: Path, chunk_dir: Path, chunk_seconds: int) -> list[AudioChunk]:
    """
    Segment the normalized audio into fixed-duration chunks using ffmpeg's
    segment muxer with `-reset_timestamps 1` so each chunk decodes cleanly
    on its own.
    """
    _require_ffmpeg()
    chunk_dir.mkdir(parents=True, exist_ok=True)
    pattern = chunk_dir / "chunk_%04d.mp3"

    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(normalized_path),
                "-f", "segment",
                "-segment_time", str(chunk_seconds),
                "-reset_timestamps", "1",
                "-c", "copy",
                str(pattern),
            ],
            capture_output=True, text=True, timeout=1800, check=True,
        )
    except subprocess.TimeoutExpired as exc:
        raise AudioProcessingError("Audio chunking timed out.") from exc
    except subprocess.CalledProcessError as exc:
        raise AudioProcessingError(f"ffmpeg failed to chunk audio: {exc.stderr.strip()}") from exc

    chunk_files = sorted(chunk_dir.glob("chunk_*.mp3"))
    if not chunk_files:
        raise AudioProcessingError("Chunking produced no output files.")

    chunks: list[AudioChunk] = []
    offset = 0.0
    for idx, path in enumerate(chunk_files):
        duration = probe_duration_seconds(path)
        chunks.append(
            AudioChunk(
                path=path,
                chunk_index=idx,
                start_offset_seconds=offset,
                end_offset_seconds=offset + duration,
            )
        )
        offset += duration

    return chunks


def enforce_chunk_size_limit(chunks: list[AudioChunk]) -> None:
    """Defensive check: even after chunking by duration, verify each chunk fits Whisper's size cap."""
    settings = get_settings()
    limit_bytes = settings.whisper_max_file_size_mb * 1024 * 1024
    for chunk in chunks:
        size = chunk.path.stat().st_size
        if size > limit_bytes:
            raise AudioProcessingError(
                f"Chunk {chunk.chunk_index} is {size / (1024 * 1024):.1f}MB, exceeding the "
                f"{settings.whisper_max_file_size_mb}MB Whisper limit. Reduce whisper_chunk_seconds in config."
            )


async def prepare_audio_chunks(raw_path: Path, work_dir: Path) -> list[AudioChunk]:
    """
    Full preprocessing pipeline, run off the event loop since ffmpeg/ffprobe
    are blocking subprocess calls.
    """
    settings = get_settings()

    def _run() -> list[AudioChunk]:
        work_dir.mkdir(parents=True, exist_ok=True)
        probe_duration_seconds(raw_path)  # validates the source is real audio; raises otherwise
        normalized_path = work_dir / "normalized.mp3"
        normalize_to_mono_mp3(raw_path, normalized_path)

        chunk_dir = work_dir / "chunks"
        chunks = split_into_chunks(normalized_path, chunk_dir, settings.whisper_chunk_seconds)
        enforce_chunk_size_limit(chunks)
        return chunks

    return await asyncio.to_thread(_run)
