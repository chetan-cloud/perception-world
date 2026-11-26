import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from perception.env import RoomEnv, seeds_for, seed_split


def test_gym_contract():
    check_env(RoomEnv(), skip_render_check=True)


def test_reproducible_trajectory():
    a, b = RoomEnv(), RoomEnv()
    oa, _ = a.reset(seed=42)
    ob, _ = b.reset(seed=42)
    np.testing.assert_array_equal(oa["image"], ob["image"])
    for _ in range(20):
        ra, rb = a.step([1, .2]), b.step([1, .2])
        np.testing.assert_array_equal(ra[0]["image"], rb[0]["image"])
        assert ra[1:] == rb[1:]


def test_generated_rooms_and_split():
    splits = [set(seeds_for(s, 100)) for s in ("train", "validation", "test")]
    assert not (splits[0] & splits[1] or splits[1] & splits[2] or splits[0] & splits[2])
    env = RoomEnv()
    for seed in range(100):
        obs, _ = env.reset(seed=seed)
        assert env.observation_space.contains(obs)
        assert not env._blocked(env.position)
        assert "position" not in obs
        assert seed_split(seed) in ("train", "validation", "test")


def test_collision_and_success():
    env = RoomEnv()
    env.reset(seed=2)
    env.position = np.array([env.radius + .001, 1.0])
    env.heading = np.pi
    before = env.position.copy()
    _, _, _, _, info = env.step([1, 0])
    assert info["collision"]
    np.testing.assert_array_equal(before, env.position)
    env.position = env.boxes[env.target] + [env.box_half + env.radius + .05, 0]
    _, _, done, _, info = env.step([0, 0])
    assert done and info["success"]
    with pytest.raises(RuntimeError):
        env.step([0, 0])


def test_timeout_and_invalid_action():
    env = RoomEnv(max_steps=1)
    env.reset(seed=3)
    with pytest.raises(ValueError):
        env.step([np.nan, 0])
    _, _, terminated, truncated, _ = env.step([0, 0])
    assert truncated and not terminated
