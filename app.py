"""
app.py
------
Gradio UI for HuggingFace Spaces deployment.
Wraps process_video() in a simple upload → process → download interface.
No annotated video option — clips + report only.
"""

from __future__ import annotations
import logging
import tempfile
import os
from pathlib import Path

import gradio as gr

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

from src.pipeline import process_video

# ---------------------------------------------------------------------------
# Gradio handler
# ---------------------------------------------------------------------------

def run_extraction(
    video_file,
    score_threshold: float,
    max_highlights: int,
    frame_stride: int,
    progress=gr.Progress(track_tqdm=True),
) -> tuple:
    """
    Gradio callback.
    Returns (clips_list, highlights_markdown, report_json).
    """
    if video_file is None:
        return [], "Please upload a video.", "{}"

    # Patch config overrides from UI sliders
    import yaml
    with open("config.yaml") as f:
        cfg_data = yaml.safe_load(f)

    cfg_data["segmenter"]["score_threshold"] = score_threshold
    cfg_data["segmenter"]["max_highlights"]  = int(max_highlights)
    cfg_data["detector"]["frame_stride"]     = int(frame_stride)
    # Never write annotated video from the app
    cfg_data["exporter"]["write_annotated"]  = False

    # Write a temporary config for this run
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False)
    yaml.safe_dump(cfg_data, tmp)
    tmp.close()

    def _cb(frac, msg):
        progress(frac, desc=msg)

    try:
        result = process_video(
            input_path=video_file,
            config_path=tmp.name,
            output_base="data/output",
            progress_callback=_cb,
        )
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass

    if result["error"]:
        return [], f"Error: {result['error']}", "{}"

    # ---- Format highlight summary using plain-English event names ----
    lines = []
    for i, h in enumerate(result["timestamps"]):
        events = " · ".join(h.get("event_types") or h["triggered_heuristics"]) or "—"
        duration = h.get("duration", round(h["end"] - h["start"], 1))
        lines.append(
            f"**[{i+1}]** &nbsp; `{h['start']:.1f}s – {h['end']:.1f}s` &nbsp;"
            f"({duration:.1f}s) &nbsp; score: `{h['peak_score']:.2f}`\n\n"
            f"&nbsp;&nbsp;&nbsp;&nbsp;{events}"
        )
    highlights_md = "\n\n---\n\n".join(lines) if lines else "_No highlights found._"

    # ---- Report JSON ----
    import json
    report_text = "{}"
    if result["report"] and Path(result["report"]).exists():
        with open(result["report"]) as f:
            report_text = f.read()

    clips = result["clips"] or []
    return clips, highlights_md, report_text


# ---------------------------------------------------------------------------
# UI layout
# ---------------------------------------------------------------------------

with gr.Blocks(
    title="Dashcam Highlight Extractor",
    theme=gr.themes.Soft(),
    css=".gradio-container {max-width: 1100px; margin: auto;}",
) as demo:

    gr.Markdown(
        """
        # Dashcam Highlight Extractor
        Upload a dashcam video clip to automatically extract dangerous or interesting moments.
        """
    )

    with gr.Row():
        with gr.Column(scale=2):
            video_input = gr.Video(label="Upload Dashcam Video", sources=["upload"])

        with gr.Column(scale=1):
            gr.Markdown("### Settings")
            score_slider = gr.Slider(
                minimum=0.1, maximum=0.9, value=0.28, step=0.05,
                label="Sensitivity",
                info="Lower = more highlights extracted",
            )
            max_highlights_slider = gr.Slider(
                minimum=1, maximum=50, value=30, step=1,
                label="Max Highlights",
            )
            stride_slider = gr.Slider(
                minimum=1, maximum=4, value=2, step=1,
                label="Speed vs Quality",
                info="Higher = faster processing, slightly lower accuracy",
            )
            run_btn = gr.Button("Extract Highlights", variant="primary")

    gr.Markdown("---")
    gr.Markdown("## Results")

    clips_gallery = gr.Gallery(
        label="Highlight Clips",
        columns=3,
        height=320,
        object_fit="contain",
    )

    highlights_md = gr.Markdown(label="Highlight Summary")
    report_json   = gr.Code(label="Full JSON Report", language="json", lines=12)

    run_btn.click(
        fn=run_extraction,
        inputs=[video_input, score_slider, max_highlights_slider, stride_slider],
        outputs=[clips_gallery, highlights_md, report_json],
    )

    gr.Markdown(
        """
        ---
        **What each event type means:**

        | Event | Description |
        |---|---|
        | Sudden Approach | Object rapidly closing distance to the dashcam vehicle |
        | Lane Cut-In | Vehicle sharply changing trajectory toward your lane |
        | Sudden Braking | Vehicle ahead decelerating sharply |
        | Close Proximity | Object unusually close for its size class |
        | Chaotic Traffic | Unusually busy scene with diverse, erratic motion |
        | Pedestrian / Cyclist Hazard | Vulnerable road user detected in the road zone |
        | Rapid Scene Change | High rate of objects entering / leaving the frame |
        | Collision Risk | Estimated time-to-collision is critically low |
        """
    )


if __name__ == "__main__":
    demo.launch(share=False)
