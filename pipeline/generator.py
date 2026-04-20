import os
from openai import OpenAI
from .tone_adapter import TONE_DESCRIPTIONS

FORMAT_PROMPTS = {
    "summary": (
        "Create a clear, structured summary with 5-7 bullet points highlighting the key insights. "
        "Each bullet should be concise and actionable. Start with a 1-sentence overview."
    ),
    "article": (
        "Write a comprehensive blog article (800-1200 words) with: "
        "1) An engaging SEO-friendly title, "
        "2) A compelling introduction (2-3 paragraphs), "
        "3) 3-4 main sections with bold subheadings, "
        "4) A conclusion with a clear call to action."
    ),
    "linkedin": (
        "Write a LinkedIn post (150-300 words) with: "
        "1) A strong hook as the first line (makes people click 'see more'), "
        "2) 3 key insights formatted with line breaks for readability, "
        "3) A thought-provoking question or CTA at the end, "
        "4) 3-5 relevant hashtags on the last line. "
        "Use emojis sparingly and strategically."
    ),
    "caption": (
        "Write a punchy social media caption (50-100 words) for Instagram/Facebook. "
        "Start with an attention-grabbing first line, deliver the core message, "
        "and end with 5-8 relevant hashtags."
    ),
    "email": (
        "Write a professional email newsletter with: "
        "Subject: [compelling subject line]\n"
        "---\n"
        "1) Warm greeting, "
        "2) Main content with 2-3 key points as short paragraphs, "
        "3) A single clear CTA button/link placeholder [CTA], "
        "4) Professional sign-off."
    ),
    "seo": (
        "Write an SEO-optimized article (1000-1500 words) with: "
        "SEO Title: [title]\nMeta Description: [150 chars max]\nTarget Keywords: [5 keywords]\n---\n"
        "1) Keyword-rich introduction, "
        "2) 4-5 sections with H2/H3 headings containing keywords, "
        "3) FAQ section (3 questions), "
        "4) Conclusion with keyword summary."
    ),
    "tweet": (
        "Create a Twitter/X thread of 6-8 tweets. Format as:\n"
        "🧵 1/N [hook tweet that makes people want to read more]\n"
        "2/N [key point 1]\n"
        "...\n"
        "N/N [summary + call to follow]\n"
        "Each tweet must be under 280 characters. Use 2-3 hashtags total in the thread."
    ),
}


def generate_formats(
    segments: dict,
    formats: list[str],
    tone: str,
    cleaned_text: str = "",
) -> dict[str, str]:
    """
    Generate multiple content formats from segmented content.
    Returns dict of {format_name: generated_content}.
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    tone_instruction = TONE_DESCRIPTIONS.get(tone, TONE_DESCRIPTIONS["formal"])
    language = segments.get("language", "the same language as the input")

    context = _build_context(segments, cleaned_text)
    results = {}

    for fmt in formats:
        if fmt not in FORMAT_PROMPTS:
            continue

        system_prompt = (
            f"You are an expert content creator. Generate high-quality content based on provided material.\n\n"
            f"TONE: {tone.upper()} — {tone_instruction}\n"
            f"LANGUAGE: Respond in {language}. Do NOT translate — keep the original language.\n"
            f"FORMAT INSTRUCTIONS: {FORMAT_PROMPTS[fmt]}"
        )

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Create a {fmt} from this content:\n\n{context}"},
            ],
            temperature=0.7,
        )

        results[fmt] = response.choices[0].message.content.strip()

    return results


def _build_context(segments: dict, cleaned_text: str) -> str:
    parts = []

    if segments.get("title"):
        parts.append(f"TITLE: {segments['title']}")
    if segments.get("overall_summary"):
        parts.append(f"SUMMARY: {segments['overall_summary']}")
    if segments.get("key_themes"):
        parts.append(f"KEY THEMES: {', '.join(segments['key_themes'])}")
    if segments.get("audience"):
        parts.append(f"TARGET AUDIENCE: {segments['audience']}")

    if segments.get("topics"):
        parts.append("\nTOPICS:")
        for topic in segments["topics"]:
            parts.append(f"\n## {topic.get('heading', 'Section')}")
            parts.append(topic.get("content", ""))
            if topic.get("key_points"):
                parts.append("Key points: " + " | ".join(topic["key_points"]))

    if cleaned_text:
        parts.append(f"\nFULL TRANSCRIPT:\n{cleaned_text}")

    return "\n".join(parts)
