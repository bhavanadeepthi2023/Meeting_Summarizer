import math
import os
import shutil
from typing import List
from pydub import AudioSegment

MAX_CHUNK_BYTES = 24 * 1024 * 1024  # 24 MB Whisper ingestion boundary


def preprocess_and_split(input_path: str, working_dir: str) -> List[str]:
    """
    Downsamples audio to 16kHz mono 64kbps and partitions into sub-24MB segments if needed.
    """
    os.makedirs(working_dir, exist_ok=True)
    audio = AudioSegment.from_file(input_path)

    # Normalize channels and sample rate to optimize file size without losing ASR accuracy
    audio = audio.set_frame_rate(16000).set_channels(1)

    normalized_file = os.path.join(working_dir, "normalized.mp3")
    audio.export(normalized_file, format="mp3", bitrate="64k")

    total_bytes = os.path.getsize(normalized_file)
    if total_bytes <= MAX_CHUNK_BYTES:
        final_path = os.path.join(working_dir, "chunk_0.mp3")
        shutil.move(normalized_file, final_path)
        return [final_path]

    num_segments = math.ceil(total_bytes / MAX_CHUNK_BYTES)
    total_ms = len(audio)
    segment_duration_ms = math.ceil(total_ms / num_segments)

    chunk_paths: List[str] = []
    for idx in range(num_segments):
        start_ms = idx * segment_duration_ms
        end_ms = min((idx + 1) * segment_duration_ms, total_ms)
        segment = audio[start_ms:end_ms]

        output_path = os.path.join(working_dir, f"chunk_{idx}.mp3")
        segment.export(output_path, format="mp3", bitrate="64k")
        chunk_paths.append(output_path)

    if os.path.exists(normalized_file):
        os.remove(normalized_file)

    return chunk_paths