"""Command-line entry point for the first real-to-simulation stage."""
import argparse
from pathlib import Path
from perception.capture.prepare import prepare
from perception.capture.segment import segment


def main():
    parser = argparse.ArgumentParser(description="Prepare real captures and segment object tracks")
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="Prepare photos or video; no GPU required")
    prep.add_argument("source", type=Path)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--fps", type=float, default=2)
    prep.add_argument("--max-frames", type=int, default=120)
    prep.add_argument("--max-side", type=int, default=1280)
    seg = commands.add_parser("segment", help="Run SAM 3 tracking; requires CUDA and model access")
    seg.add_argument("scene", type=Path)
    seg.add_argument("--output", type=Path, required=True)
    seg.add_argument("--prompt", required=True)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            result = prepare(args.source, args.output, sample_fps=args.fps, max_frames=args.max_frames, max_side=args.max_side)
            print(f"Prepared {len(result['frames'])} frames: {args.output / 'contact-sheet.jpg'}")
        else:
            result = segment(args.scene, args.output, args.prompt)
            print(f"Exported {len(result['objects'])} tracks: {args.output / 'objects.json'}")
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(2, f"Error: {exc}\n")

if __name__ == "__main__":
    main()
