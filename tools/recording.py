# Copyright (C) 2026 World Agents Corporation.
from __future__ import annotations

import argparse
import json
import shutil
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import gymnasium as gym
import h5py
import imageio.v2 as imageio
import numpy as np
import torch

import robofactory  # noqa: F401 - registers RoboFactory environments
from mani_skill.utils import common
from robofactory.utils.wrappers.record import RecordEpisodeMA


@contextmanager
def isolated_legacy_numpy_rng(seed: int):
    """Seed legacy NumPy users during single-Env construction/reset, then restore."""

    numpy_state = np.random.get_state()
    try:
        np.random.seed(seed)
        yield
    finally:
        np.random.set_state(numpy_state)


def normalize_depth_observation(observation):
    """Convert ManiSkill signed millimeter depth to uint16 with zero invalid."""

    for sensor in observation.get("sensor_data", {}).values():
        if "depth" not in sensor:
            continue
        depth = sensor["depth"]
        if isinstance(depth, torch.Tensor):
            sensor["depth"] = torch.clamp(depth, min=0).to(torch.uint16)
        else:
            sensor["depth"] = np.clip(depth, 0, None).astype(np.uint16)
    return observation


class DepthUint16Wrapper(gym.Wrapper):
    """Normalize RGB-D observations before RecordEpisodeMA buffers them."""

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        return normalize_depth_observation(observation), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return (
            normalize_depth_observation(observation),
            reward,
            terminated,
            truncated,
            info,
        )


def annotate_depth_datasets(replay_h5: Path, cameras: tuple[str, ...]) -> None:
    with h5py.File(replay_h5, "r+") as replay:
        trajectory = replay[sorted(replay.keys())[0]]
        for camera in cameras:
            depth = trajectory[f"obs/sensor_data/{camera}/depth"]
            depth.attrs["unit"] = "millimeter"
            depth.attrs["invalid_value"] = np.uint16(0)


def as_numpy(value):
    value = np.asarray(common.to_numpy(value))
    return value[0] if value.ndim > 1 and value.shape[0] == 1 else value


def success_value(info) -> bool:
    value = info.get("success", False)
    return bool(as_numpy(value).reshape(-1)[0])


def write_rgb_videos(
    replay_h5: Path, output_dir: Path, cameras: tuple[str, ...], fps: int = 20
) -> dict[str, int]:
    counts = {}
    with h5py.File(replay_h5, "r") as replay:
        trajectory = replay[sorted(replay.keys())[0]]
        for camera in cameras:
            frames = trajectory[f"obs/sensor_data/{camera}/rgb"]
            with imageio.get_writer(
                output_dir / f"{camera}.mp4",
                fps=fps,
                codec="libx264",
                quality=8,
                macro_block_size=1,
            ) as writer:
                for frame in frames:
                    writer.append_data(frame)
            counts[camera] = int(len(frames))
    return counts


def depth_dataset_metrics(dataset: h5py.Dataset, chunk_size: int = 16) -> dict:
    minimum = np.inf
    maximum = -np.inf
    valid_pixels = 0
    total_pixels = 0
    for start in range(0, len(dataset), chunk_size):
        values = dataset[start : start + chunk_size]
        finite = np.isfinite(values)
        valid = finite & (values > 0)
        valid_pixels += int(valid.sum())
        total_pixels += int(values.size)
        if finite.any():
            finite_values = values[finite]
            minimum = min(minimum, float(finite_values.min()))
            maximum = max(maximum, float(finite_values.max()))
    return {
        "frames": int(len(dataset)),
        "shape": list(dataset.shape),
        "dtype": str(dataset.dtype),
        "unit": dataset.attrs.get("unit", "unknown"),
        "invalid_value": int(dataset.attrs.get("invalid_value", 0)),
        "minimum": minimum if np.isfinite(minimum) else None,
        "maximum": maximum if np.isfinite(maximum) else None,
        "valid_pixel_ratio": valid_pixels / total_pixels if total_pixels else 0.0,
    }


def sensor_metrics(replay_h5: Path, cameras: tuple[str, ...]) -> tuple[dict, dict, dict]:
    camera_motion = {}
    depth = {}
    parameters = {}
    with h5py.File(replay_h5, "r") as replay:
        trajectory = replay[sorted(replay.keys())[0]]
        for camera in cameras:
            rgb = trajectory[f"obs/sensor_data/{camera}/rgb"]
            depth_dataset = trajectory[f"obs/sensor_data/{camera}/depth"]
            parameter_group = trajectory[f"obs/sensor_param/{camera}"]
            poses = parameter_group["cam2world_gl"][:]
            translations = poses[:, :3, 3]
            camera_motion[camera] = {
                "frames": int(len(rgb)),
                "translation_path_span_m": float(np.linalg.norm(np.ptp(translations, axis=0))),
                "first_last_translation_m": float(np.linalg.norm(translations[-1] - translations[0])),
                "first_last_rgb_mean_abs_delta": float(
                    np.abs(rgb[-1].astype(np.float32) - rgb[0].astype(np.float32)).mean()
                ),
            }
            depth[camera] = depth_dataset_metrics(depth_dataset)
            parameters[camera] = {
                name: {"frames": int(len(dataset)), "shape": list(dataset.shape), "dtype": str(dataset.dtype)}
                for name, dataset in parameter_group.items()
            }
    return camera_motion, depth, parameters
