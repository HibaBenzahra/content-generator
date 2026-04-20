import os
from openai import OpenAI

TONE_DESCRIPTIONS = {
    "formal": (
        "Use professional, structured language. Avoid contractions and colloquialisms. "
        "Maintain an authoritative, respectful tone suitable for corporate or academic contexts."
    ),
    "casual": (
        "Use friendly, conversational language. Contractions are welcome. "
        "Keep it relatable, approachable, and easy to read as if talking to a friend."
    ),
    "marketing": (
        "Use persuasive, benefit-driven language. Highlight value propositions, "
        "create urgency, include a clear call-to-action. Focus on 'what's in it for the reader'."
    ),
    "educational": (
        "Use clear, informative language. Break down complex ideas step-by-step. "
        "Use analogies, examples, and a teaching tone. Prioritize clarity over flair."
    ),
    "inspirational": (
        "Use emotionally resonant, motivating language. Tell stories, evoke feelings, "
        "empower the reader. Be uplifting and forward-looking with a vision-driven voice."
    ),
}


def adapt_tone(content: str, tone: str, format_type: str) -> str:
    """Re-adapt existing generated content to a specific tone."""
    if tone not in TONE_DESCRIPTIONS:
        raise ValueError(f"Unknown tone '{tone}'. Choose from: {list(TONE_DESCRIPTIONS.keys())}")

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    tone_instruction = TONE_DESCRIPTIONS[tone]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    f"You are a tone adaptation specialist. Rewrite the provided {format_type} "
                    f"content while preserving all information and structure.\n\n"
                    f"Target tone: {tone.upper()}\n{tone_instruction}\n\n"
                    "Preserve the original language. Return only the rewritten content."
                ),
            },
            {"role": "user", "content": content},
        ],
        temperature=0.5,
    )

    return response.choices[0].message.content.strip()
