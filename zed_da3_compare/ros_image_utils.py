"""Minimal ROS Image <-> numpy helpers for DA3 depth inference."""

from __future__ import annotations

import cv2
import numpy as np
from sensor_msgs.msg import Image


def image_msg_to_rgb(msg: Image) -> np.ndarray:
    """Convert a ROS color Image message to an HxWx3 uint8 RGB array."""
    enc = msg.encoding.lower()
    data = np.frombuffer(msg.data, dtype=np.uint8)

    if enc == "rgb8":
        row = msg.width * 3
        return data.reshape(msg.height, msg.step)[:, :row].reshape(msg.height, msg.width, 3).copy()

    if enc == "bgr8":
        row = msg.width * 3
        bgr = data.reshape(msg.height, msg.step)[:, :row].reshape(msg.height, msg.width, 3)
        return bgr[:, :, ::-1].copy()

    if enc == "rgba8":
        row = msg.width * 4
        rgba = data.reshape(msg.height, msg.step)[:, :row].reshape(msg.height, msg.width, 4)
        return rgba[:, :, :3].copy()

    if enc == "bgra8":
        row = msg.width * 4
        bgra = data.reshape(msg.height, msg.step)[:, :row].reshape(msg.height, msg.width, 4)
        return bgra[:, :, [2, 1, 0]].copy()

    raise ValueError(f"Unsupported color image encoding: {msg.encoding}")


def depth_msg_to_meters(msg: Image) -> np.ndarray:
    """Convert a ROS depth Image message to an HxW float32 depth map in metres."""
    enc = msg.encoding.lower()

    if enc == "32fc1":
        data = np.frombuffer(msg.data, dtype=np.float32)
        row_values = msg.step // 4
        return data.reshape(msg.height, row_values)[:, : msg.width].copy()

    if enc == "16uc1":
        data = np.frombuffer(msg.data, dtype=np.uint16)
        row_values = msg.step // 2
        depth_mm = data.reshape(msg.height, row_values)[:, : msg.width]
        return (depth_mm.astype(np.float32) * 0.001).copy()

    raise ValueError(f"Unsupported depth image encoding: {msg.encoding}")


def depth_to_image_msg(depth_m: np.ndarray, src_msg: Image) -> Image:
    """Create a 32FC1 ROS Image message from an HxW float32 depth map in metres."""
    depth_m = np.ascontiguousarray(depth_m, dtype=np.float32)

    msg = Image()
    msg.header = src_msg.header
    msg.height, msg.width = depth_m.shape[:2]
    msg.encoding = "32FC1"
    msg.is_bigendian = False
    msg.step = int(msg.width * 4)
    msg.data = depth_m.tobytes()
    return msg


def depth_to_preview_msg(
    depth_m: np.ndarray,
    src_msg: Image,
    *,
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
    use_inverse_depth: bool = True,
) -> Image:
    """Create an rgb8 preview image normalized by percentiles for human inspection."""
    # El preview no pretende ser métrico; solo quiere mostrar estructura visual.
    # Por eso:
    # - ignoramos valores inválidos,
    # - usamos percentiles para no dejar que unos pocos outliers dominen la escala,
    # - y opcionalmente invertimos la profundidad, porque a menudo es más útil
    #   ver "cerca = más brillante" en un vistazo rápido.
    depth_m = np.asarray(depth_m, dtype=np.float32)
    valid = np.isfinite(depth_m) & (depth_m > 0.0)

    preview = np.zeros(depth_m.shape, dtype=np.uint8)
    if valid.any():
        values = depth_m[valid]
        if use_inverse_depth:
            # Convertimos depth -> inverse depth porque, visualmente, suele dar
            # más contraste en las zonas cercanas y ayuda a leer mejor la escena.
            values = 1.0 / np.maximum(values, 1e-6)

        lo = float(np.percentile(values, low_percentile))
        hi = float(np.percentile(values, high_percentile))
        if hi <= lo:
            hi = lo + 1e-6

        transformed = np.zeros_like(depth_m, dtype=np.float32)
        transformed[valid] = 1.0 / np.maximum(depth_m[valid], 1e-6) if use_inverse_depth else depth_m[valid]
        transformed = np.clip((transformed - lo) / (hi - lo), 0.0, 1.0)
        preview = (transformed * 255.0).astype(np.uint8)

    # Convertimos el monocromo normalizado en una imagen en color para que el
    # preview sea más fácil de leer en RViz, rqt_image_view o herramientas similares.
    colored_bgr = cv2.applyColorMap(preview, cv2.COLORMAP_TURBO)
    colored_rgb = colored_bgr[:, :, ::-1]

    msg = Image()
    msg.header = src_msg.header
    msg.height, msg.width = colored_rgb.shape[:2]
    msg.encoding = "rgb8"
    msg.is_bigendian = False
    msg.step = int(msg.width * 3)
    msg.data = np.ascontiguousarray(colored_rgb).tobytes()
    return msg
