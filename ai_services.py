import json
import os
from faster_whisper import WhisperModel
from openai import OpenAI
from schemas import MeetingSummaryResult

# Initialize Faster-Whisper on CPU with int8 quantization
whisper_model = WhisperModel("base", device="cpu", compute_type="int8")

# Initialize Groq client
groq_client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=os.environ.get("GROQ_API_KEY", ""),
)

SYSTEM_PROMPT = """You are an automated technical documentation and meeting intelligence system.
Analyze the provided meeting transcript with high factual fidelity.
You MUST output valid JSON matching this exact schema:
{
  "meeting_title": "string",
  "executive_summary": "string",
  "key_topics": ["string"],
  "decisions": [{"decision": "string", "rationale": "string or null"}],
  "action_items": [
    {
      "task": "string",
      "assignee": "string or null",
      "deadline": "string or null",
      "priority": "low" | "medium" | "high" | "critical"
    }
  ]
}
Do not include any conversational preamble or markdown code blocks, only raw JSON.
"""


def resolve_active_model() -> str:
    """Queries Groq API dynamically and selects the best active model available on the key."""
    try:
        response = groq_client.models.list()
        available_ids = [m.id for m in response.data]

        preferences = [
            "llama-3.3-70b-versatile",
            "llama-3.3-70b-specdec",
            "llama3-70b-8192",
            "llama3-8b-8192",
            "gemma2-9b-it",
            "mixtral-8x7b-32768",
        ]
        for candidate in preferences:
            if candidate in available_ids:
                return candidate

        # Fallback to the first available model if preferred list is not found
        if available_ids:
            return available_ids[0]
    except Exception:
        pass

    return "llama3-8b-8192"


def transcribe_audio_segments(segment_paths: list[str]) -> str:
    """Runs local Whisper on each audio file."""
    transcripts = []
    for path in segment_paths:
        segments, _ = whisper_model.transcribe(path, beam_size=5)
        text = " ".join([segment.text for segment in segments])
        transcripts.append(text)
    return "\n".join(transcripts)


def extract_structured_meeting_data(transcript: str) -> MeetingSummaryResult:
    """Extracts structured schema using the dynamically resolved active model."""
    selected_model = resolve_active_model()

    response = groq_client.chat.completions.create(
        model=selected_model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Transcript:\n\n{transcript}"},
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
    )
    raw_content = response.choices[0].message.content
    parsed_json = json.loads(raw_content)
    return MeetingSummaryResult.model_validate(parsed_json)