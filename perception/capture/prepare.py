"""Normalize photographs or sample video into a reconstruction-ready frame sequence."""
import hashlib
import json
import math
import re
import shutil
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}


def natural_key(path):
    return [int(s) if s.isdigit() else s.lower() for s in re.split(r"(\d+)", path.name)]


def quality(image):
    gray = np.asarray(image.convert("L"), dtype=np.float32)
    dx, dy = np.diff(gray, axis=1), np.diff(gray, axis=0)
    # A relative sharpness diagnostic, not a calibrated blur detector.
    return float((np.mean(dx * dx) + np.mean(dy * dy)) / 2)


def prepare(source, output, *, sample_fps=2.0, max_frames=120, max_side=1280):
    source, output = Path(source).resolve(), Path(output).resolve()
    if not source.exists():
        raise ValueError(f"Input does not exist: {source}")
    if not math.isfinite(sample_fps) or sample_fps <= 0 or max_frames < 1 or max_side < 16:
        raise ValueError("fps must be finite and positive; max_frames >= 1; max_side >= 16")
    if output.exists():
        raise ValueError("Output already exists; choose a new scene directory")
    if source.is_dir() and (output == source or source in output.parents):
        raise ValueError("Output must be outside the input directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".capture-", dir=output.parent))
    records, dimensions = [], None

    def save(image, origin, timestamp):
        nonlocal dimensions
        image = ImageOps.exif_transpose(image).convert("RGB")
        original_size = list(image.size)
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        if min(image.size) < 2:
            raise ValueError("Frame dimensions must be at least 2 pixels")
        if dimensions is not None and image.size != dimensions:
            raise ValueError("Frames must have a consistent size after orientation and resizing")
        dimensions = image.size
        name = f"{len(records):06d}.jpg"
        target = stage / "images" / name
        image.save(target, quality=95)
        records.append({"index": len(records), "file": f"images/{name}", "source_name": origin,
                        "timestamp_seconds": timestamp, "original_size": original_size,
                        "size": list(image.size), "sharpness_score": quality(image),
                        "sha256": hashlib.sha256(target.read_bytes()).hexdigest()})

    try:
        (stage / "images").mkdir()
        if source.is_dir() or source.suffix.lower() in IMAGE_SUFFIXES:
            paths = sorted((p for p in source.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES), key=natural_key) if source.is_dir() else [source]
            if not paths:
                raise ValueError("No supported images found")
            # Cover the sequence, rather than keeping only its first frames.
            indices = np.linspace(0, len(paths) - 1, min(max_frames, len(paths)), dtype=int)
            for i in indices:
                with Image.open(paths[i]) as image:
                    save(image, paths[i].name, None)
            kind, effective_fps = "images", None
        else:
            try:
                import cv2
            except ImportError as exc:
                raise RuntimeError("Video input requires: pip install -e '.[capture]'") from exc
            cap = cv2.VideoCapture(str(source))
            try:
                fps = cap.get(cv2.CAP_PROP_FPS)
                if not cap.isOpened() or not math.isfinite(fps) or fps <= 0:
                    raise ValueError("Cannot read video or determine its frame rate")
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total < 1:
                    raise ValueError("Video has no usable frame count")
                # Sampling spans the whole clip and remains bounded in memory.
                stride = max(1, int(math.ceil(fps / sample_fps)))
                candidates = range(0, total, stride)
                positions = np.linspace(0, len(candidates) - 1, min(max_frames, len(candidates)), dtype=int)
                for position in positions:
                    index = candidates[int(position)]
                    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                    ok, bgr = cap.read()
                    if not ok:
                        raise ValueError(f"Failed to decode video frame {index}")
                    save(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), source.name, index / fps)
                kind, effective_fps = "video", fps
            finally:
                cap.release()
        manifest = {"schema_version": 1, "status": "prepared", "input_kind": kind,
                    "source_name": source.name, "source_fps": effective_fps,
                    "requested_sample_fps": sample_fps, "frames": records,
                    "geometry": {"status": "not_estimated", "scale_meters": None},
                    "segmentation": {"status": "not_run"}}
        (stage / "scene.json").write_text(json.dumps(manifest, indent=2) + "\n")
        # Bounded contact sheet for checking coverage and blur before GPU work.
        sheet = Image.new("RGB", (640, math.ceil(min(12, len(records)) / 4) * 120), "#18202c")
        for slot, index in enumerate(np.linspace(0, len(records)-1, min(12, len(records)), dtype=int)):
            with Image.open(stage / records[index]["file"]) as image:
                image.thumbnail((156, 116))
                sheet.paste(image, ((slot % 4)*160, (slot // 4)*120))
        sheet.save(stage / "contact-sheet.jpg")
        stage.rename(output)
        return manifest
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
