# Step 1: prepare captures and segment objects

This stage converts real photos/video into an ordered image sequence, then optionally runs SAM 3 to segment and track a prompted object category. It does not yet estimate cameras, reconstruct 3D geometry, or assign physical properties.

## Prepare media (works on Mac, no model download)

```sh
pip install -e '.[capture,dev]'
perception-capture prepare /path/to/video.mp4 --output runs/captures/kitchen --fps 2 --max-frames 120
# Alternatively: a single photograph or a directory of overlapping photographs.
perception-capture prepare /path/to/photos --output runs/captures/table
```

Outputs:

- `images/000000.jpg`, etc.: EXIF-oriented, RGB, aspect-preserving resized frames. Filenames start at zero for compatibility with SAM 3 and future reconstruction adapters.
- `scene.json`: input names, original/prepared dimensions, timestamps for video, frame hashes, and relative sharpness diagnostics. No absolute source paths or EXIF metadata are copied into the export.
- `contact-sheet.jpg`: coverage preview before expensive GPU processing.

Existing output directories are refused. A failed run removes its temporary output. Photo directories use natural filename order; rename files to encode capture order if needed. When capped, samples span the whole input sequence. Video timestamps are calculated from the decoder's reported FPS and are approximate for variable-frame-rate media. This initial sampler limits frame count/resolution; it does not guarantee visual overlap, remove moving objects, estimate calibrated blur, or select geometric keyframes.

Use a slow translating camera, overlapping viewpoints, good light, and a mostly static scene. A stationary panoramic rotation does not provide the same geometry constraints as translating through a room. Start with roughly 10–30 selected frames to keep reconstruction experiments manageable. One known physical dimension will be needed later to validate metric scale.

## Segment and track with SAM 3 (CUDA machine)

Use the official [SAM 3 setup instructions](https://github.com/facebookresearch/sam3#installation) in a separate Python 3.12+ environment with compatible PyTorch/CUDA. Request model access and authenticate with Hugging Face as described upstream. Install this project into that same environment:

```sh
pip install -e '.[capture]'
perception-capture segment runs/captures/kitchen --prompt cup --output runs/segments/cups
perception-capture segment runs/captures/kitchen --prompt chair --output runs/segments/chairs
```

The adapter uses the documented session/prompt/propagation API. It has not been exercised with SAM 3 weights on a CUDA GPU in this project yet. Its surrounding exporter is tested against explicit fixture outputs; these tests are not evidence of model segmentation accuracy. The current local development machine is an ARM Mac, so it can run ingestion/export tests but not this CUDA backend. SAM 3 is an optional external dependency with its own license and weight-access requirements; weights are not bundled.

Choose one concept per run. SAM assigns instance IDs within that prompt session. IDs from separate prompts/runs are not merged or deduplicated. They are not guaranteed 3D identities; tracking errors and ID switches require inspection. Newly visible objects depend on model detection behavior. There is no promise to discover every object or part automatically.

Outputs:

- `objects.json`: prompt, persistent session-scoped IDs, per-frame scores, pixel bounding boxes, and mask/crop references. Geometry and physical characteristics explicitly remain unknown.
- `objects/object-00007/masks/000000.png`: full prepared-frame-size binary mask. Overlapping objects have separate masks rather than overwriting one label map.
- `objects/object-00007/crops/000000.png`: bounding-box crop with transparency. Use the stored bounding box to map back into the frame.
- `overlays/000000.jpg`: inspect the masks and IDs against the input.

A missing frame, changed input hash, or invalid output aborts the export. A complete run with no detections is valid and has an empty object list. Scores are model outputs, not calibrated probabilities of physical correctness.

## Next stage

Run a VGGT camera/depth adapter on these same prepared images, reconcile its preprocessing coordinates with the masks, and fuse observations into object-level geometry. Camera estimation, multi-view mask association, metric scale calibration, collision meshes, and simulation are not implemented in this step.

Source media, frames, and segmentation outputs belong under ignored `runs/` directories. Do not commit personal footage to the public repository.
