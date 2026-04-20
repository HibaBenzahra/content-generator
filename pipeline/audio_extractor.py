import os
import tempfile
import ffmpeg


SUPPORTED_VIDEO = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv"}
SUPPORTED_AUDIO = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac"}


def extract_audio(input_path: str, output_path: str = None) -> str:
    """
    Extract audio from a video or pass through an audio file.
    Returns path to the audio file (wav format).
    """
    ext = os.path.splitext(input_path)[1].lower()

    if ext in SUPPORTED_AUDIO:
        return input_path

    if ext not in SUPPORTED_VIDEO:
        raise ValueError(f"Unsupported file format: {ext}")

    if output_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        output_path = tmp.name
        tmp.close()

    try:
        (
            ffmpeg
            .input(input_path)
            .output(
                output_path,
                acodec="pcm_s16le",
                ac=1,
                ar="16000",
                vn=None,
            )
            .overwrite_output()
            .run(quiet=True)
        )
    except ffmpeg.Error as e:
        raise RuntimeError(f"FFmpeg extraction failed: {e.stderr.decode() if e.stderr else str(e)}")

    return output_path


def is_video(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in SUPPORTED_VIDEO


def is_audio(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in SUPPORTED_AUDIO
