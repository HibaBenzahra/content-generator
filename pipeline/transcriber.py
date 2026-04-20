import os
from openai import OpenAI


def transcribe(audio_path: str, language: str = None) -> str:
    """
    Transcribe audio using OpenAI Whisper API.
    Supports FR, EN, and Moroccan dialect (auto-detected or specified).
    language: ISO-639-1 code e.g. 'fr', 'en', 'ar', or None for auto-detect.
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    with open(audio_path, "rb") as audio_file:
        kwargs = {
            "model": "whisper-1",
            "file": audio_file,
            "response_format": "text",
        }
        if language and language != "auto":
            kwargs["language"] = language

        transcript = client.audio.transcriptions.create(**kwargs)

    return transcript if isinstance(transcript, str) else transcript.text
