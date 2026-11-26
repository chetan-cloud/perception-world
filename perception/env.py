"""Planar navigation with a perspective ray-cast camera; no rigid-body physics."""
import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

COLORS = np.array([[230, 70, 65], [65, 200, 110], [65, 120, 240]], dtype=np.uint8)
NAMES = ("red_box", "green_box", "blue_box")


def seed_split(seed):
    """Disjoint deterministic 80/10/10 assignment by room seed."""
    bucket = int(seed) % 10
    return "test" if bucket == 0 else "validation" if bucket == 1 else "train"


def seeds_for(split, count):
    if split not in ("train", "validation", "test") or count < 1:
        raise ValueError("Use train, validation, or test and a positive count")
    result, seed = [], 0
    while len(result) < count:
        if seed_split(seed) == split:
            result.append(seed)
        seed += 1
    return result


class RoomEnv(gym.Env):
    """RGB + target identity in; normalized forward/turn command out.

    Exact positions are deliberately absent from observations and info.
    debug_state() exposes privileged state only for visualization and tests.
    """
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}
    radius = 0.18
    box_half = 0.3

    def __init__(self, image_size=64, max_steps=300, render_mode="rgb_array"):
        super().__init__()
        if image_size < 16 or max_steps < 1:
            raise ValueError("image_size >= 16 and max_steps >= 1 required")
        if render_mode != "rgb_array":
            raise ValueError("Only rgb_array is supported")
        self.image_size, self.max_steps = image_size, max_steps
        self.render_mode = render_mode
        self.action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.observation_space = spaces.Dict({
            "image": spaces.Box(0, 255, (image_size, image_size, 3), np.uint8),
            "target": spaces.Discrete(3),
        })
        self._ready = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.room_size = float(self.np_random.uniform(5, 8))
        # Clearance exceeds the robot diameter; all boxes have traversable gaps.
        points = []
        for _ in range(4):
            for attempt in range(10000):
                point = self.np_random.uniform(0.85, self.room_size - 0.85, 2)
                if all(np.linalg.norm(point - p) > 1.35 for p in points):
                    points.append(point)
                    break
            else:
                raise RuntimeError("Could not sample a room")
        self.boxes = np.array(points[:3])
        self.position = points[3].copy()
        self.heading = float(self.np_random.uniform(-math.pi, math.pi))
        self.target = int(self.np_random.integers(3))
        self.steps, self.collisions = 0, 0
        self._ready, self._done = True, False
        return self._observe(), {"target_name": NAMES[self.target]}

    def _blocked(self, point):
        if np.any(point < self.radius) or np.any(point > self.room_size - self.radius):
            return True
        delta = np.maximum(np.abs(self.boxes - point) - self.box_half, 0)
        return bool(np.any(np.linalg.norm(delta, axis=1) <= self.radius))

    def step(self, action):
        if not self._ready or self._done:
            raise RuntimeError("Call reset before stepping a new episode")
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (2,) or not np.all(np.isfinite(action)):
            raise ValueError("Expected two finite action values")
        action = np.clip(action, -1, 1)
        self.heading = float((self.heading + action[1] * 0.16 + math.pi) % (2 * math.pi) - math.pi)
        direction = np.array([math.cos(self.heading), math.sin(self.heading)])
        collision = False
        # Small substeps prevent tunneling through box corners.
        for _ in range(4):
            candidate = self.position + direction * float(action[0]) * 0.025
            if self._blocked(candidate):
                collision = True
                break
            self.position = candidate
        self.steps += 1
        self.collisions += int(collision)
        distance = np.linalg.norm(np.maximum(np.abs(self.position - self.boxes[self.target]) - self.box_half, 0))
        success = bool(distance < self.radius + 0.2)
        terminated = success
        truncated = self.steps >= self.max_steps and not success
        self._done = terminated or truncated
        reward = float(success) - 0.01 - 0.05 * collision
        return self._observe(), reward, terminated, truncated, {
            "success": success, "collision": collision, "steps": self.steps,
            "collisions": self.collisions, "target_name": NAMES[self.target],
        }

    def _observe(self):
        return {"image": self.render(), "target": self.target}

    def debug_state(self):
        return {"room_size": self.room_size, "position": self.position.copy(),
                "heading": self.heading, "boxes": self.boxes.copy(), "target": self.target}

    def render(self):
        if not self._ready:
            raise RuntimeError("Call reset before render")
        n = self.image_size
        offsets = np.arctan(np.linspace(-math.tan(math.pi / 6), math.tan(math.pi / 6), n))
        angles = self.heading + offsets
        rays = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        safe = np.where(np.abs(rays) < 1e-9, 1e-9, rays)
        wall_t = np.where(safe > 0, (self.room_size - self.position) / safe, -self.position / safe)
        depths = wall_t.min(axis=1)
        ids = np.full(n, -1)
        for i, center in enumerate(self.boxes):
            t1 = (center - self.box_half - self.position) / safe
            t2 = (center + self.box_half - self.position) / safe
            near = np.minimum(t1, t2).max(axis=1)
            far = np.maximum(t1, t2).min(axis=1)
            hit = (far >= np.maximum(near, 0)) & (near > 0) & (near < depths)
            depths[hit], ids[hit] = near[hit], i
        corrected = np.maximum(depths * np.cos(offsets), 0.05)
        image = np.empty((n, n, 3), np.uint8)
        image[: n // 2] = [37, 45, 60]
        image[n // 2 :] = [70, 65, 58]
        for x in range(n):
            height = min(n * 4, int(n * 0.7 / corrected[x]))
            top, bottom = max(0, n // 2 - height // 2), min(n, n // 2 + height // 2 + 1)
            color = COLORS[ids[x]] if ids[x] >= 0 else np.array([165, 170, 180])
            shade = max(0.3, 1 / (1 + corrected[x] * 0.12))
            image[top:bottom, x] = (color * shade).astype(np.uint8)
        return image
