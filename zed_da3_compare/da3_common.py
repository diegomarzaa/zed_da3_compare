"""Shared helpers for Depth Anything 3 ROS nodes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo


def load_da3_model(node: Node):
    """Load a local DA3 model using this node's model_dir and device parameters."""
    model_dir = Path(str(node.get_parameter("model_dir").value)).expanduser()
    if not str(model_dir) or not model_dir.is_dir():
        raise ValueError(
            "Parameter 'model_dir' must be a local model directory, "
            "for example: /models/da3/DA3NESTED-GIANT-LARGE-1.1"
        )

    from depth_anything_3.api import DepthAnything3

    device = str(node.get_parameter("device").value)
    model = DepthAnything3.from_pretrained(str(model_dir)).to(device).eval()
    node.get_logger().info(f"Loaded DA3 model from: {model_dir}")
    return model


def log_depth_stats(logger, label: str, prediction, depth_m: np.ndarray) -> None:
    """Log a compact range summary for a predicted depth map."""
    valid = np.isfinite(depth_m) & (depth_m > 0.0)
    if not valid.any():
        logger.warn(f"{label} depth stats: no valid positive finite values")
        return

    values = depth_m[valid]
    logger.info(
        f"{label} depth stats "
        f"is_metric={getattr(prediction, 'is_metric', None)} "
        f"scale_factor={getattr(prediction, 'scale_factor', None)} "
        f"min={np.min(values):.3f} "
        f"p01={np.percentile(values, 1):.3f} "
        f"p50={np.percentile(values, 50):.3f} "
        f"p99={np.percentile(values, 99):.3f} "
        f"max={np.max(values):.3f}"
    )


def camera_info_to_intrinsics(msg: CameraInfo) -> np.ndarray:
    """Convert ROS CameraInfo.K into a DA3 3x3 intrinsics matrix."""
    return np.asarray(msg.k, dtype=np.float32).reshape(3, 3)


def baseline_from_right_camera_info(msg: CameraInfo) -> float | None:
    """Infer rectified stereo baseline from right CameraInfo.P when available."""
    fx = float(msg.p[0])
    tx = float(msg.p[3])
    if abs(fx) < 1e-6 or abs(tx) < 1e-9:
        return None
    return -tx / fx


def stereo_extrinsics_from_baseline(baseline_m: float) -> np.ndarray:
    """Build world-to-camera extrinsics for a rectified left/right stereo pair."""
    extrinsics = np.repeat(np.eye(4, dtype=np.float32)[None, :, :], 2, axis=0)
    extrinsics[1, 0, 3] = -float(baseline_m)
    return extrinsics


def stamp_to_seconds(msg) -> float:
    """Convert a ROS message header stamp to seconds."""
    return float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
