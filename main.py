import os
import tempfile

import gradio as gr
from dotenv import load_dotenv

from pipeline import (
    extract_audio,
    transcribe,
    clean_text,
    segment_content,
    generate_formats,
    adapt_tone,
    TONE_DESCRIPTIONS,
    generate_image,
)

load_dotenv()

ALL_FORMATS = ["summary", "article", "linkedin", "caption", "email", "seo", "tweet"]
ALL_TONES = list(TONE_DESCRIPTIONS.keys())

FORMAT_LABELS = {
    "summary": "Summary",
    "article": "Blog Article",
    "linkedin": "LinkedIn Post",
    "caption": "Social Caption",
    "email": "Email Newsletter",
    "seo": "SEO Article",
    "tweet": "Tweet Thread",
}


def run_pipeline(
    audio_file,
    text_input,
    language,
    selected_formats,
    tone,
    progress=gr.Progress(track_tqdm=True),
):
    if not os.getenv("OPENAI_API_KEY"):
        raise gr.Error("OPENAI_API_KEY is not set. Add it to your .env file.")

    if not selected_formats:
        raise gr.Error("Please select at least one output format.")

    raw_text = ""
    transcript_output = ""
    cleaned_output = ""
    tmp_audio_path = None

    try:
        # Step 1 & 2: Extract audio + transcribe
        if audio_file is not None:
            progress(0.1, desc="Extracting audio...")
            audio_path = extract_audio(audio_file)
            tmp_audio_path = audio_path if audio_path != audio_file else None

            progress(0.25, desc="Transcribing with Whisper...")
            lang_code = None if language == "Auto-detect" else {
                "French": "fr", "English": "en", "Arabic / Darija": "ar"
            }.get(language)
            raw_text = transcribe(audio_path, language=lang_code)
            transcript_output = raw_text

        elif text_input and text_input.strip():
            raw_text = text_input.strip()
            transcript_output = "(Text input — no transcription needed)"
        else:
            raise gr.Error("Please upload an audio/video file or paste text.")

        # Step 3: Clean
        progress(0.40, desc="Cleaning text...")
        cleaned_text = clean_text(raw_text)
        cleaned_output = cleaned_text

        # Step 4: Segment
        progress(0.55, desc="Segmenting content...")
        segments = segment_content(cleaned_text)

        # Step 5 & 6: Generate formats with tone
        progress(0.70, desc=f"Generating {len(selected_formats)} format(s) with {tone} tone...")
        results = generate_formats(segments, selected_formats, tone, cleaned_text)

        progress(1.0, desc="Done!")

    finally:
        if tmp_audio_path and os.path.exists(tmp_audio_path):
            os.unlink(tmp_audio_path)

    # Build tab outputs — return value for each format tab
    outputs = [transcript_output, cleaned_output]
    for fmt in ALL_FORMATS:
        outputs.append(results.get(fmt, "_Not selected or not generated._"))

    # Store segments and cleaned_text in state for image generation
    outputs.append({"segments": segments, "cleaned_text": cleaned_text})

    return outputs


def run_image_generation(pipeline_state, progress=gr.Progress(track_tqdm=True)):
    import requests

    if not pipeline_state:
        raise gr.Error("Run the content pipeline first before generating an image.")

    if not os.getenv("OPENAI_API_KEY"):
        raise gr.Error("OPENAI_API_KEY is not set. Add it to your .env file.")

    progress(0.2, desc="Building image prompt...")
    segments = pipeline_state["segments"]
    cleaned_text = pipeline_state["cleaned_text"]

    progress(0.6, desc="Generating image with DALL-E 2...")
    image_url, image_prompt = generate_image(segments, cleaned_text)

    progress(0.9, desc="Downloading image...")
    response = requests.get(image_url, timeout=30)
    response.raise_for_status()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
    tmp.write(response.content)
    tmp.close()

    progress(1.0, desc="Done!")
    return tmp.name, f"**Prompt used:** {image_prompt}"


def build_ui():
    with gr.Blocks(title="Content Generator") as demo:

        gr.Markdown(
            """
            # Content Generator
            **AI-powered pipeline** — Upload audio/video or paste text, choose formats and tone, generate.
            Supports French · English · Moroccan Darija
            """,
            elem_classes="header",
        )

        pipeline_state = gr.State(None)

        with gr.Row():
            # ── Left column: inputs ──────────────────────────────────────────
            with gr.Column(scale=1):
                gr.Markdown("### Input")

                with gr.Tabs():
                    with gr.TabItem("Audio / Video File"):
                        audio_file = gr.File(
                            label="Upload file",
                            file_types=[
                                ".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac",
                                ".mp4", ".mkv", ".avi", ".mov", ".webm",
                            ],
                        )
                        language = gr.Dropdown(
                            choices=["Auto-detect", "French", "English", "Arabic / Darija"],
                            value="Auto-detect",
                            label="Transcription language",
                        )

                    with gr.TabItem("Paste Text"):
                        text_input = gr.Textbox(
                            label="Paste transcript or text",
                            placeholder="Paste your text here...",
                            lines=10,
                        )

                gr.Markdown("### Output Formats")
                format_checks = gr.CheckboxGroup(
                    choices=[(v, k) for k, v in FORMAT_LABELS.items()],
                    value=["summary", "article", "linkedin"],
                    label="Select formats to generate",
                    elem_classes="format-checks",
                )

                gr.Markdown("### Tone")
                tone_selector = gr.Radio(
                    choices=ALL_TONES,
                    value="formal",
                    label="Content tone",
                )

                generate_btn = gr.Button("Generate Content", variant="primary", size="lg")

            # ── Right column: outputs ────────────────────────────────────────
            with gr.Column(scale=1):
                gr.Markdown("### Output")

                with gr.Tabs():
                    with gr.TabItem("Transcript"):
                        out_transcript = gr.Textbox(label="Raw transcript", lines=12)
                    with gr.TabItem("Cleaned Text"):
                        out_cleaned = gr.Textbox(label="Cleaned text", lines=12)
                    with gr.TabItem(FORMAT_LABELS["summary"]):
                        out_summary = gr.Markdown()
                    with gr.TabItem(FORMAT_LABELS["article"]):
                        out_article = gr.Markdown()
                    with gr.TabItem(FORMAT_LABELS["linkedin"]):
                        out_linkedin = gr.Markdown()
                    with gr.TabItem(FORMAT_LABELS["caption"]):
                        out_caption = gr.Markdown()
                    with gr.TabItem(FORMAT_LABELS["email"]):
                        out_email = gr.Markdown()
                    with gr.TabItem(FORMAT_LABELS["seo"]):
                        out_seo = gr.Markdown()
                    with gr.TabItem(FORMAT_LABELS["tweet"]):
                        out_tweet = gr.Markdown()
                    with gr.TabItem("Image"):
                        image_btn = gr.Button("Generate Image", variant="secondary")
                        out_image = gr.Image(label="Generated Image", type="filepath")
                        out_image_prompt = gr.Markdown()

        generate_btn.click(
            fn=run_pipeline,
            inputs=[audio_file, text_input, language, format_checks, tone_selector],
            outputs=[
                out_transcript, out_cleaned,
                out_summary, out_article, out_linkedin,
                out_caption, out_email, out_seo, out_tweet,
                pipeline_state,
            ],
            show_progress=True,
        )

        image_btn.click(
            fn=run_image_generation,
            inputs=[pipeline_state],
            outputs=[out_image, out_image_prompt],
            show_progress=True,
        )

        gr.Markdown(
            "_Powered by OpenAI Whisper + GPT-4o + DALL-E 2 · Built with Gradio_",
            elem_classes="header",
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", 7860)),
        show_error=True,
        theme=gr.themes.Soft(primary_hue="violet", secondary_hue="indigo"),
        inbrowser=True,
    )
