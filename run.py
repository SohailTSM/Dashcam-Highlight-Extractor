"""
run.py
------
CLI entry point for the Dashcam Highlight Extractor.

Usage:
    python run.py --input data/input/clip.mp4
    python run.py --input data/input/clip.mp4 --config config.yaml --output data/output
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

from src.pipeline import process_video


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dashcam Highlight Extractor")
    p.add_argument("--input",          required=True, help="Path to input dashcam video")
    p.add_argument("--config",         default="config.yaml", help="Path to config.yaml")
    p.add_argument("--output",         default="data/output", help="Base output directory")
    p.add_argument("--max-highlights", type=int,   default=None, help="Max highlights to extract (overrides config)")
    p.add_argument("--threshold",      type=float, default=None, help="Score threshold 0-1 (overrides config)")
    p.add_argument("--stride",         type=int,   default=None, help="Frame stride (overrides config)")
    p.add_argument("--annotated",      action="store_true",     help="Write full annotated video (slow, off by default)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not Path(args.input).exists():
        print(f"[ERROR] Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Apply CLI overrides on top of config
    import yaml
    with open(args.config) as f:
        cfg_data = yaml.safe_load(f)

    if args.max_highlights is not None:
        cfg_data["segmenter"]["max_highlights"] = args.max_highlights
    if args.threshold is not None:
        cfg_data["segmenter"]["score_threshold"] = args.threshold
    if args.stride is not None:
        cfg_data["detector"]["frame_stride"] = args.stride
    if args.annotated:
        cfg_data["exporter"]["write_annotated"] = True

    # Write patched config to a temp file for this run
    import tempfile, os
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    yaml.safe_dump(cfg_data, tmp)
    tmp.close()
    effective_config = tmp.name

    print(f"\n🎬 Processing: {args.input}")
    print(f"   threshold={cfg_data['segmenter']['score_threshold']}  "
          f"max_highlights={cfg_data['segmenter']['max_highlights']}  "
          f"stride={cfg_data['detector']['frame_stride']}")
    print("─" * 50)

    result = process_video(
        input_path=args.input,
        config_path=effective_config,
        output_base=args.output,
    )

    # Clean up temp config
    try:
        os.unlink(effective_config)
    except Exception:
        pass

    if result["error"]:
        print(f"\n❌ Error: {result['error']}", file=sys.stderr)
        sys.exit(1)

    highlights = result["timestamps"]
    print(f"\n✅ Found {len(highlights)} highlight(s):\n")

    for i, h in enumerate(highlights):
        events = ", ".join(h.get("event_types") or h["triggered_heuristics"]) or "—"
        print(f"  [{i+1}] {h['start']:.1f}s – {h['end']:.1f}s  peak={h['peak_score']:.2f}")
        print(f"       Events: {events}")
        if h.get("clip"):
            print(f"       Clip  : {h['clip']}")

    if result.get("annotated_video"):
        print(f"\n📹 Annotated video : {result['annotated_video']}")
    if result["report"]:
        print(f"📄 Report          : {result['report']}")
    print()


if __name__ == "__main__":
    main()
