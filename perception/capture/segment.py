"""SAM 3 video adapter and model-independent object export.

Masks remain in prepared-image pixel coordinates. No 3D identity or physics
is inferred here. Track IDs are scoped to one prompt/session.
"""
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def sam3_outputs(images, prompt):
    try:
        import torch
        from sam3.model_builder import build_sam3_video_predictor
    except ImportError as exc:
        raise RuntimeError("Install SAM 3 in its Python 3.12+ CUDA environment; see capture documentation") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("The SAM 3 backend requires a CUDA GPU; media preparation works without one")
    predictor = build_sam3_video_predictor()
    session = None
    try:
        session = predictor.handle_request(request={"type": "start_session", "resource_path": str(images)})["session_id"]
        initial = predictor.handle_request(request={"type": "add_prompt", "session_id": session, "frame_index": 0, "text": prompt})
        yield 0, initial["outputs"]
        for result in predictor.handle_stream_request(request={"type": "propagate_in_video", "session_id": session}):
            yield int(result["frame_index"]), result["outputs"]
    finally:
        if session is not None:
            predictor.handle_request(request={"type": "close_session", "session_id": session})


def segment(scene, output, prompt, *, outputs=None):
    scene, output = Path(scene).resolve(), Path(output).resolve()
    if not prompt.strip():
        raise ValueError("Provide a nonempty object concept, such as 'cup'")
    if output.exists():
        raise ValueError("Output already exists; choose a new directory")
    manifest = json.loads((scene / "scene.json").read_text())
    frames = manifest["frames"]
    if not frames:
        raise ValueError("Scene contains no frames")
    # Verify alignment has not changed since capture preparation.
    for frame in frames:
        path = (scene / frame["file"]).resolve()
        if not path.is_relative_to(scene) or hashlib.sha256(path.read_bytes()).hexdigest() != frame["sha256"]:
            raise ValueError("Frame path or contents do not match the scene manifest")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".segment-", dir=output.parent))
    tracks, seen = {}, set()
    supplied_outputs = outputs is not None
    try:
        (stage / "overlays").mkdir()
        iterator = outputs if supplied_outputs else sam3_outputs(scene / "images", prompt)
        try:
            for index, prediction in iterator:
                if index in seen:
                    continue  # SAM may emit prompted frame again during propagation.
                if not 0 <= index < len(frames):
                    raise ValueError("Model returned an out-of-range frame index")
                seen.add(index)
                ids = array(prediction["out_obj_ids"]).reshape(-1)
                masks = array(prediction["out_binary_masks"])
                scores = array(prediction["out_probs"]).reshape(-1)
                if not len(ids) == len(masks) == len(scores) or len(set(map(int, ids))) != len(ids):
                    raise ValueError("Model returned inconsistent object outputs")
                with Image.open(scene / frames[index]["file"]) as raw:
                    image = raw.convert("RGB")
                overlay = np.asarray(image).copy()
                for object_id, mask, score in zip(ids, masks, scores):
                    object_id, score = int(object_id), float(score)
                    if object_id < 0 or not np.isfinite(score) or not 0 <= score <= 1:
                        raise ValueError("Invalid object ID or score")
                    mask = np.asarray(mask)
                    while mask.ndim > 2 and mask.shape[0] == 1:
                        mask = mask[0]
                    if mask.ndim != 2 or not np.all(np.isfinite(mask)):
                        raise ValueError("Expected a finite 2D object mask")
                    binary = Image.fromarray((mask > .5).astype(np.uint8) * 255).resize(image.size, Image.Resampling.NEAREST)
                    bbox = binary.getbbox()
                    if bbox is None:
                        continue
                    key = f"object-{object_id:05d}"
                    directory = stage / "objects" / key
                    (directory / "masks").mkdir(parents=True, exist_ok=True)
                    (directory / "crops").mkdir(exist_ok=True)
                    name = f"{index:06d}.png"
                    binary.save(directory / "masks" / name)
                    cutout = image.convert("RGBA")
                    cutout.putalpha(binary)
                    cutout.crop(bbox).save(directory / "crops" / name)
                    track = tracks.setdefault(key, {"id": key, "prompt": prompt, "observations": [],
                        "physical_properties": {"status": "unknown"}, "geometry": {"status": "not_reconstructed"}})
                    track["observations"].append({"frame_index": index, "bbox_xyxy": list(bbox), "score": score,
                        "mask": f"objects/{key}/masks/{name}", "crop": f"objects/{key}/crops/{name}"})
                    active = np.asarray(binary) > 0
                    color = np.random.default_rng(object_id).integers(60, 240, 3)
                    overlay[active] = (.55 * overlay[active] + .45 * color).astype(np.uint8)
                canvas = Image.fromarray(overlay)
                draw = ImageDraw.Draw(canvas)
                for key, track in tracks.items():
                    for record in track["observations"][-1:]:
                        if record["frame_index"] == index:
                            draw.rectangle(record["bbox_xyxy"], outline="white", width=2)
                            draw.text(tuple(record["bbox_xyxy"][:2]), key, fill="white")
                canvas.save(stage / "overlays" / f"{index:06d}.jpg")
        finally:
            if hasattr(iterator, "close"):
                iterator.close()
        if seen != set(range(len(frames))):
            raise ValueError("Model did not return every prepared frame; refusing incomplete output")
        result = {"schema_version": 1, "backend": "external_test_outputs" if supplied_outputs else "sam3",
            "scene_manifest_sha256": hashlib.sha256((scene / "scene.json").read_bytes()).hexdigest(),
            "prompt": prompt, "frame_count": len(frames), "identity_scope": "single_prompt_session_not_verified_3d_identity",
            "objects": list(tracks.values())}
        (stage / "objects.json").write_text(json.dumps(result, indent=2) + "\n")
        stage.rename(output)
        return result
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
