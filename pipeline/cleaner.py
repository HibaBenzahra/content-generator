import os
from openai import OpenAI

_SYSTEM_PROMPT = """You are a professional text editor specialized in cleaning raw transcripts.

Your tasks:
1. Remove spoken fillers: "um", "uh", "you know", "like", "euh", "bref", "voilà", "enfin", "hein", "quoi", "walu", "wach", "daba", and similar filler words in French, English, or Moroccan Darija.
2. Fix punctuation and capitalization.
3. Split run-on sentences logically.
4. Preserve the original meaning and language — do NOT translate.
5. Keep proper nouns, technical terms, and code-switching between languages intact.
6. Return only the cleaned text, no explanations.
"""


def clean_text(raw_text: str) -> str:
    """Remove fillers, fix punctuation, and clean raw transcript text."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Clean this transcript:\n\n{raw_text}"},
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content.strip()
