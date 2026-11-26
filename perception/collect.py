"""Collect camera/action sequences for offline prediction experiments."""
import argparse
import json
from pathlib import Path
import numpy as np
from perception.env import RoomEnv, seeds_for


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--split", choices=["train", "validation", "test"], default="train")
    parser.add_argument("--output", type=Path, default=Path("runs/data"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for seed in seeds_for(args.split, args.episodes):
        env = RoomEnv()
        obs, _ = env.reset(seed=seed)
        env.action_space.seed(seed)
        images, actions, rewards, endings, timeouts = [obs["image"]], [], [], [], []
        for _ in range(env.max_steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            images.append(obs["image"])
            actions.append(action)
            rewards.append(reward)
            endings.append(terminated)
            timeouts.append(truncated)
            if terminated or truncated:
                break
        np.savez_compressed(args.output / f"room-{seed}.npz", images=np.stack(images),
                            actions=np.asarray(actions, dtype=np.float32), rewards=np.asarray(rewards, dtype=np.float32),
                            terminated=np.asarray(endings, dtype=bool), truncated=np.asarray(timeouts, dtype=bool),
                            target=np.int64(obs["target"]))
        (args.output / f"room-{seed}.json").write_text(json.dumps({"seed": seed, "split": args.split,
            "generator_version": "0.1.0", "policy": "random", "steps": len(actions)}, indent=2) + "\n")
        env.close()

if __name__ == "__main__":
    main()
