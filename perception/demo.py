"""Export a camera + privileged overhead replay without opening a window."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from perception.env import RoomEnv, COLORS, NAMES, seeds_for


def frame(env, observation):
    state = env.debug_state()
    panel = Image.new("RGB", (640, 360), "#111827")
    panel.paste(Image.fromarray(observation["image"]).resize((320, 320)), (0, 32))
    draw = ImageDraw.Draw(panel)
    draw.text((12, 10), "ROBOT CAMERA | target: " + NAMES[env.target], fill="white")
    draw.text((335, 10), "DEBUG MAP (not model input)", fill="white")
    scale = 280 / state["room_size"]
    def xy(p):
        return 340 + p[0] * scale, 50 + p[1] * scale
    draw.rectangle((340, 50, 620, 330), outline="white", width=2)
    for i, box in enumerate(state["boxes"]):
        a, b = xy(box - env.box_half), xy(box + env.box_half)
        draw.rectangle((*a, *b), fill=tuple(int(v) for v in COLORS[i]), outline="white" if i == env.target else None, width=3)
    x, y = xy(state["position"])
    draw.ellipse((x-5, y-5, x+5, y+5), fill="white")
    draw.line((x, y, x + 16*np.cos(state["heading"]), y + 16*np.sin(state["heading"])), fill="white", width=2)
    return panel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--split", choices=["train", "validation", "test"], default="test")
    parser.add_argument("--output", type=Path, default=Path("runs/demo"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for seed in seeds_for(args.split, args.episodes):
        env = RoomEnv()
        obs, info = env.reset(seed=seed)
        rng = np.random.default_rng(seed + 100000)
        frames, latencies = [frame(env, obs)], []
        total = 0
        for step in range(env.max_steps):
            action = np.array([0.8, np.sin(step / 18) * 0.8 + rng.uniform(-0.2, 0.2)], np.float32)
            start = time.perf_counter()
            obs, reward, terminated, truncated, info = env.step(action)
            latencies.append((time.perf_counter() - start) * 1000)
            total += reward
            if step % 3 == 0 or terminated or truncated:
                frames.append(frame(env, obs))
            if terminated or truncated:
                break
        frames[0].save(args.output / f"room-{seed}.gif", save_all=True, append_images=frames[1:], duration=100, loop=0)
        records.append({"seed": seed, "split": args.split, "policy": "scripted_exploration_not_trained", "return": total, **info,
                        "environment_step_ms_p50": float(np.percentile(latencies, 50)),
                        "environment_step_ms_p95": float(np.percentile(latencies, 95))})
        env.close()
    (args.output / "metrics.json").write_text(json.dumps(records, indent=2) + "\n")
    print(json.dumps(records, indent=2))

if __name__ == "__main__":
    main()
