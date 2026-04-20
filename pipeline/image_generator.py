import os
import requests
from openai import OpenAI


def generate_image(segments: dict, cleaned_text: str = "") -> tuple[str, str]:
    """
    Generate an image based on content segments.
    Returns (image_url, image_prompt).
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    title = segments.get("title", "")
    summary = segments.get("overall_summary", "")
    themes = ", ".join(segments.get("key_themes", []))

    context = f"Title: {title}\nSummary: {summary}\nThemes: {themes}"

    prompt_response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You create concise DALL-E image prompts. "
                    "Given content context, write a single descriptive image prompt (max 200 chars) "
                    "suitable for a professional blog or social media post. "
                    "Focus on visual elements, mood, and composition. No text in the image."
                ),
            },
            {"role": "user", "content": f"Create an image prompt for this content:\n{context}"},
        ],
        temperature=0.7,
        max_tokens=100,
    )

    image_prompt = prompt_response.choices[0].message.content.strip()

    image_response = client.images.generate(
        model="dall-e-2",
        prompt=image_prompt,
        size="512x512",
        n=1,
    )

    image_url = image_response.data[0].url
    return image_url, image_prompt
