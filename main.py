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

FORMAT_ICONS = {
    "summary": "📋",
    "article": "📝",
    "linkedin": "💼",
    "caption": "📸",
    "email": "✉️",
    "seo": "🔍",
    "tweet": "🐦",
}

TONE_ICONS = {
    "formal": "🎩",
    "casual": "😊",
    "marketing": "📣",
    "educational": "🎓",
    "inspirational": "✨",
}

CSS = """
/* ── Base & Reset ─────────────────────────────────────────── */
:root {
    --lime: #84cc16;
    --lime-bright: #a3e635;
    --lime-dim: #1a2e05;
    --lime-glow: rgba(132, 204, 22, 0.15);
    --bg-0: #0a0a0a;
    --bg-1: #111111;
    --bg-2: #181818;
    --bg-3: #222222;
    --border: #2a2a2a;
    --border-accent: #84cc16;
    --text-1: #f0f0f0;
    --text-2: #a0a0a0;
    --text-3: #555555;
    --radius: 10px;
}

body, .gradio-container {
    background-color: var(--bg-0) !important;
    font-family: 'Inter', 'Segoe UI', system-ui, sans-serif !important;
}

.gradio-container {
    max-width: 1400px !important;
    margin: 0 auto !important;
}

/* ── Gradio overrides ─────────────────────────────────────── */
.dark, [data-testid] {
    --body-background-fill: var(--bg-0) !important;
    --background-fill-primary: var(--bg-1) !important;
    --background-fill-secondary: var(--bg-2) !important;
    --border-color-primary: var(--border) !important;
    --color-accent: var(--lime) !important;
    --body-text-color: var(--text-1) !important;
    --block-label-text-color: var(--text-2) !important;
    --input-background-fill: var(--bg-2) !important;
    --input-border-color: var(--border) !important;
}

/* ── Header ───────────────────────────────────────────────── */
.app-header {
    background: linear-gradient(135deg, #0f1a03 0%, #111111 50%, #0a1a0a 100%);
    border: 1px solid var(--border);
    border-bottom: 2px solid var(--lime);
    border-radius: var(--radius);
    padding: 28px 36px;
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
}

.app-header::before {
    content: '';
    position: absolute;
    top: -40px; right: -40px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(132,204,22,0.08) 0%, transparent 70%);
    pointer-events: none;
}

.app-header h1 {
    font-size: 1.9rem !important;
    font-weight: 700 !important;
    color: var(--lime-bright) !important;
    margin: 0 0 6px 0 !important;
    letter-spacing: -0.5px;
    line-height: 1.2 !important;
}

.app-header p {
    color: var(--text-2) !important;
    font-size: 0.88rem !important;
    margin: 0 !important;
    line-height: 1.5 !important;
}

.lang-badge {
    display: inline-block;
    background: var(--lime-dim);
    color: var(--lime-bright);
    border: 1px solid rgba(132,204,22,0.3);
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.75rem;
    font-weight: 600;
    margin-top: 10px;
    letter-spacing: 0.5px;
}

/* ── Section labels ───────────────────────────────────────── */
.section-label {
    color: var(--lime) !important;
    font-size: 0.7rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 1.5px !important;
    margin-bottom: 10px !important;
    padding-bottom: 6px !important;
    border-bottom: 1px solid var(--border) !important;
}

/* ── Panels ───────────────────────────────────────────────── */
.panel {
    background: var(--bg-1) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 20px !important;
}

.panel-output {
    background: var(--bg-1) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
}

/* ── Inputs ───────────────────────────────────────────────── */
.gr-input, textarea, input[type="text"] {
    background: var(--bg-2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-1) !important;
    border-radius: 8px !important;
}

.gr-input:focus, textarea:focus {
    border-color: var(--lime) !important;
    box-shadow: 0 0 0 2px var(--lime-glow) !important;
}

/* ── Tabs ─────────────────────────────────────────────────── */
.tab-nav button {
    background: transparent !important;
    color: var(--text-2) !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    padding: 8px 14px !important;
    transition: color 0.2s, border-color 0.2s !important;
}

.tab-nav button:hover {
    color: var(--text-1) !important;
    background: var(--bg-2) !important;
}

.tab-nav button.selected {
    color: var(--lime) !important;
    border-bottom-color: var(--lime) !important;
    background: transparent !important;
    font-weight: 600 !important;
}

/* ── Buttons ──────────────────────────────────────────────── */
.btn-generate {
    background: var(--lime) !important;
    color: #0a0a0a !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    padding: 14px !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: background 0.2s, transform 0.1s, box-shadow 0.2s !important;
    letter-spacing: 0.3px;
}

.btn-generate:hover {
    background: var(--lime-bright) !important;
    box-shadow: 0 4px 20px rgba(132,204,22,0.3) !important;
    transform: translateY(-1px) !important;
}

.btn-generate:active {
    transform: translateY(0) !important;
}

.btn-image {
    background: transparent !important;
    color: var(--lime) !important;
    border: 1px solid var(--lime) !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 10px !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: background 0.2s, box-shadow 0.2s !important;
}

.btn-image:hover {
    background: var(--lime-glow) !important;
    box-shadow: 0 0 12px rgba(132,204,22,0.2) !important;
}

/* ── Checkboxes ───────────────────────────────────────────── */
.format-checks .wrap {
    display: grid !important;
    grid-template-columns: 1fr 1fr !important;
    gap: 6px !important;
}

.format-checks label {
    background: var(--bg-2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 8px !important;
    padding: 8px 12px !important;
    color: var(--text-2) !important;
    font-size: 0.83rem !important;
    cursor: pointer !important;
    transition: border-color 0.2s, color 0.2s, background 0.2s !important;
}

.format-checks label:hover {
    border-color: rgba(132,204,22,0.4) !important;
    color: var(--text-1) !important;
}

.format-checks input[type="checkbox"]:checked + label,
.format-checks label:has(input:checked) {
    border-color: var(--lime) !important;
    color: var(--lime) !important;
    background: var(--lime-dim) !important;
}

/* ── Radio (tone) ─────────────────────────────────────────── */
.tone-radio .wrap {
    display: flex !important;
    flex-wrap: wrap !important;
    gap: 6px !important;
}

.tone-radio label {
    background: var(--bg-2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 20px !important;
    padding: 5px 14px !important;
    color: var(--text-2) !important;
    font-size: 0.82rem !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
}

.tone-radio label:hover {
    border-color: rgba(132,204,22,0.4) !important;
    color: var(--text-1) !important;
}

.tone-radio input[type="radio"]:checked + label,
.tone-radio label:has(input:checked) {
    border-color: var(--lime) !important;
    color: var(--lime) !important;
    background: var(--lime-dim) !important;
    font-weight: 600 !important;
}

/* ── File upload ──────────────────────────────────────────── */
.upload-area {
    border: 1.5px dashed var(--border) !important;
    border-radius: 10px !important;
    background: var(--bg-2) !important;
    transition: border-color 0.2s !important;
}

.upload-area:hover {
    border-color: rgba(132,204,22,0.4) !important;
}

/* ── Textbox outputs ──────────────────────────────────────── */
.output-text textarea {
    background: var(--bg-2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-1) !important;
    font-size: 0.88rem !important;
    line-height: 1.65 !important;
    border-radius: 8px !important;
}

/* ── Markdown output ──────────────────────────────────────── */
.prose, .gr-prose, [class*="prose"] {
    color: var(--text-1) !important;
    font-size: 0.88rem !important;
    line-height: 1.7 !important;
}

.prose h1, .prose h2, .prose h3 {
    color: var(--lime-bright) !important;
    font-weight: 600 !important;
}

.prose strong { color: var(--lime-bright) !important; }
.prose ul li::marker { color: var(--lime) !important; }
.prose a { color: var(--lime) !important; }

/* ── Dropdown ─────────────────────────────────────────────── */
select, .gr-dropdown {
    background: var(--bg-2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-1) !important;
    border-radius: 8px !important;
}

/* ── Block labels ─────────────────────────────────────────── */
.block-label, label span {
    color: var(--text-2) !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
}

/* ── Progress bar ─────────────────────────────────────────── */
.progress-bar { background: var(--lime) !important; }

/* ── Footer ───────────────────────────────────────────────── */
.app-footer {
    text-align: center;
    color: var(--text-3) !important;
    font-size: 0.75rem !important;
    margin-top: 16px !important;
    padding: 10px !important;
    border-top: 1px solid var(--border) !important;
}

.app-footer a { color: var(--lime) !important; }

/* ── Scrollbar ────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-1); }
::-webkit-scrollbar-thumb { background: var(--bg-3); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--lime-dim); }

/* ── Image output ─────────────────────────────────────────── */
.gr-image { border-radius: 10px !important; border: 1px solid var(--border) !important; }
"""


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

        progress(0.40, desc="Cleaning text...")
        cleaned_text = clean_text(raw_text)
        cleaned_output = cleaned_text

        progress(0.55, desc="Segmenting content...")
        segments = segment_content(cleaned_text)

        progress(0.70, desc=f"Generating {len(selected_formats)} format(s) with {tone} tone...")
        results = generate_formats(segments, selected_formats, tone, cleaned_text)

        progress(1.0, desc="Done!")

    finally:
        if tmp_audio_path and os.path.exists(tmp_audio_path):
            os.unlink(tmp_audio_path)

    outputs = [transcript_output, cleaned_output]
    for fmt in ALL_FORMATS:
        outputs.append(results.get(fmt, "_Not selected or not generated._"))

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


THEME = gr.themes.Base(
        primary_hue=gr.themes.colors.lime,
        secondary_hue=gr.themes.colors.green,
        neutral_hue=gr.themes.colors.zinc,
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    ).set(
        body_background_fill="#0a0a0a",
        body_background_fill_dark="#0a0a0a",
        block_background_fill="#111111",
        block_background_fill_dark="#111111",
        block_border_color="#2a2a2a",
        block_border_color_dark="#2a2a2a",
        input_background_fill="#181818",
        input_background_fill_dark="#181818",
        input_border_color="#2a2a2a",
        input_border_color_dark="#2a2a2a",
        button_primary_background_fill="#84cc16",
        button_primary_background_fill_dark="#84cc16",
        button_primary_text_color="#0a0a0a",
        button_primary_text_color_dark="#0a0a0a",
        button_secondary_background_fill="#181818",
        button_secondary_background_fill_dark="#181818",
        button_secondary_text_color="#84cc16",
        button_secondary_text_color_dark="#84cc16",
        button_secondary_border_color="#84cc16",
        button_secondary_border_color_dark="#84cc16",
    )


def build_ui():
    with gr.Blocks(title="Content Generator") as demo:

        gr.HTML("""
        <div class="app-header">
            <h1>⚡ Content Generator</h1>
            <p>AI-powered pipeline — upload audio/video or paste text, pick your formats and tone, generate instantly.</p>
            <span class="lang-badge">🌐 FR · EN · Moroccan Darija</span>
        </div>
        """)

        pipeline_state = gr.State(None)

        with gr.Row(equal_height=False):

            # ── Left: Inputs ──────────────────────────────────────
            with gr.Column(scale=4, min_width=340):

                gr.HTML('<div class="section-label">Input</div>')

                with gr.Tabs(elem_classes="input-tabs"):
                    with gr.TabItem("🎵 Audio / Video"):
                        audio_file = gr.File(
                            label="Upload file",
                            file_types=[
                                ".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac",
                                ".mp4", ".mkv", ".avi", ".mov", ".webm",
                            ],
                            elem_classes="upload-area",
                        )
                        language = gr.Dropdown(
                            choices=["Auto-detect", "French", "English", "Arabic / Darija"],
                            value="Auto-detect",
                            label="Transcription Language",
                        )

                    with gr.TabItem("✏️ Paste Text"):
                        text_input = gr.Textbox(
                            label="Transcript or text",
                            placeholder="Paste your content here...",
                            lines=9,
                            elem_classes="output-text",
                        )

                gr.HTML('<div class="section-label" style="margin-top:18px">Output Formats</div>')
                format_checks = gr.CheckboxGroup(
                    choices=[(f"{FORMAT_ICONS[k]}  {v}", k) for k, v in FORMAT_LABELS.items()],
                    value=["summary", "article", "linkedin"],
                    label="",
                    elem_classes="format-checks",
                )

                gr.HTML('<div class="section-label" style="margin-top:16px">Tone</div>')
                tone_selector = gr.Radio(
                    choices=[f"{TONE_ICONS[t]}  {t.capitalize()}" for t in ALL_TONES],
                    value=f"{TONE_ICONS['formal']}  Formal",
                    label="",
                    elem_classes="tone-radio",
                )

                gr.HTML('<div style="margin-top:20px"></div>')
                generate_btn = gr.Button(
                    "⚡ Generate Content",
                    variant="primary",
                    elem_classes="btn-generate",
                )

            # ── Right: Outputs ─────────────────────────────────────
            with gr.Column(scale=6, min_width=480, elem_classes="panel-output"):

                gr.HTML('<div class="section-label" style="padding:16px 16px 0">Output</div>')

                with gr.Tabs(elem_classes="output-tabs"):
                    with gr.TabItem("📄 Transcript"):
                        out_transcript = gr.Textbox(
                            label="", lines=18, 
                            elem_classes="output-text",
                            placeholder="Transcript will appear here after processing...",
                        )
                    with gr.TabItem("🧹 Cleaned"):
                        out_cleaned = gr.Textbox(
                            label="", lines=18, 
                            elem_classes="output-text",
                            placeholder="Cleaned text will appear here...",
                        )
                    with gr.TabItem("📋 Summary"):
                        out_summary = gr.Markdown(value="_Run the pipeline to generate a summary._")
                    with gr.TabItem("📝 Article"):
                        out_article = gr.Markdown(value="_Run the pipeline to generate a blog article._")
                    with gr.TabItem("💼 LinkedIn"):
                        out_linkedin = gr.Markdown(value="_Run the pipeline to generate a LinkedIn post._")
                    with gr.TabItem("📸 Caption"):
                        out_caption = gr.Markdown(value="_Run the pipeline to generate a social caption._")
                    with gr.TabItem("✉️ Email"):
                        out_email = gr.Markdown(value="_Run the pipeline to generate an email newsletter._")
                    with gr.TabItem("🔍 SEO"):
                        out_seo = gr.Markdown(value="_Run the pipeline to generate an SEO article._")
                    with gr.TabItem("🐦 Tweets"):
                        out_tweet = gr.Markdown(value="_Run the pipeline to generate a tweet thread._")
                    with gr.TabItem("🖼️ Image"):
                        gr.HTML('<div style="padding:12px 0 8px;color:#555;font-size:0.8rem;">Generate a visual for your content after running the pipeline.</div>')
                        image_btn = gr.Button(
                            "🎨 Generate Image",
                            variant="secondary",
                            elem_classes="btn-image",
                        )
                        out_image = gr.Image(
                            label="", type="filepath",
                            elem_classes="gr-image",
                        )
                        out_image_prompt = gr.Markdown()

        gr.HTML('<div class="app-footer">Powered by OpenAI Whisper · GPT-4o · DALL-E 2 &nbsp;|&nbsp; Built with <a href="https://gradio.app" target="_blank">Gradio</a></div>')

        def run_pipeline_wrapper(audio_file, text_input, language, selected_formats, tone_with_icon, progress=gr.Progress(track_tqdm=True)):
            tone = tone_with_icon.split("  ")[-1].lower() if tone_with_icon else "formal"
            fmt_keys = selected_formats
            return run_pipeline(audio_file, text_input, language, fmt_keys, tone, progress)

        generate_btn.click(
            fn=run_pipeline_wrapper,
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

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", 8080)),
        show_error=True,
        inbrowser=True,
        theme=THEME,
        css=CSS,
    )
