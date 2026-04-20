import os
import json
from openai import OpenAI

_SYSTEM_PROMPT = """You are a content strategist and analyst.

Given a cleaned transcript or text, extract and return a structured JSON object with:
{
  "title": "A concise title for the main topic",
  "language": "detected language (e.g. 'French', 'English', 'Arabic/Darija', 'Mixed')",
  "audience": "likely target audience",
  "key_themes": ["theme1", "theme2", "theme3"],
  "topics": [
    {
      "heading": "Topic/Section heading",
      "content": "Relevant excerpt or summary of this section",
      "key_points": ["key point 1", "key point 2"]
    }
  ],
  "overall_summary": "2-3 sentence summary of the entire content",
  "tone_detected": "detected tone (e.g. conversational, professional, educational)"
}

Return ONLY valid JSON. No markdown, no explanation.
"""


def segment_content(cleaned_text: str) -> dict:
    """Identify topics, structure, and key themes from cleaned text."""
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Analyze and segment this content:\n\n{cleaned_text}"},
        ],
        temperature=0.3,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content.strip()
    return json.loads(raw)
