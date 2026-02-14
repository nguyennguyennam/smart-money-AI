import os
import gradio as gr
from faster_whisper import WhisperModel



MODEL_SIZE = os.getenv("WHISPER_MODEL", "small")
DEVICE = os.getenv("WHISPER_DEVICE", "cpu")
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE", "int8")

# Load model once
model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)


def transcribe(audio_path: str, language: str, task: str):
    if audio_path is None:
        return "Please record audio first."

    lang = language.strip() if language else None
    if lang == "":
        lang = None  # auto detect

    segments, info = model.transcribe(
        audio_path,
        language=lang,
        task=task,
        vad_filter=True,
        beam_size=5
    )

    text = "".join(seg.text for seg in segments).strip()

    detected = getattr(info, "language", "unknown")
    prob = getattr(info, "language_probability", None)

    meta = f"Detected: {detected}"
    if prob:
        meta += f" (p={prob:.2f})"

    if not text:
        return f"{meta}\n\n(No speech detected.)"

    return f"{meta}\n\n{text}"


with gr.Blocks(title="Speech-to-Text (Whisper)") as demo:
    gr.Markdown(
        """
        # 🎙️ Speech-to-Text (Whisper + Gradio)
        - Click **Record** to speak
        - Click **Transcribe** to extract text
        """
    )

    with gr.Row():
        audio = gr.Audio(
            sources=["microphone"],
            type="filepath",
            label="Record from microphone"
        )

    with gr.Row():
        language = gr.Dropdown(
            choices=["", "en", "vi"],
            value="en",   
            label="Language (leave blank for auto-detect)"
        )

        task = gr.Radio(
            choices=["transcribe", "translate"],
            value="transcribe",
            label="Mode"
        )

    btn = gr.Button("📝 Transcribe")
    output = gr.Textbox(label="Result", lines=10)

    btn.click(
        fn=transcribe,
        inputs=[audio, language, task],
        outputs=output
    )

    gr.Markdown(
        """
        **Notes**
        - `translate` converts speech to English.
        - Change model via environment variable:
          `WHISPER_MODEL=tiny` (faster) or `medium` (more accurate).
        """
    )

if __name__ == "__main__":
    demo.launch()
