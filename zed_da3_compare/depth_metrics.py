"""Depth metrics for DA3/ZED comparison workflows."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np


def resize_depth_to_shape(depth_m: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resize depth to HxW using linear interpolation."""
    import cv2

    height, width = shape
    if depth_m.shape[:2] == (height, width):
        return depth_m
    return cv2.resize(depth_m, (width, height), interpolation=cv2.INTER_LINEAR)


def valid_depth_mask(
    depth_m: np.ndarray,
    *,
    min_depth_m: float = 0.2,
    max_depth_m: float = 20.0,
) -> np.ndarray:
    """Return valid finite positive depth pixels inside the requested range."""
    return (
        np.isfinite(depth_m)
        & (depth_m >= float(min_depth_m))
        & (depth_m <= float(max_depth_m))
    )


def joint_valid_mask(
    *depth_maps: np.ndarray,
    min_depth_m: float = 0.2,
    max_depth_m: float = 20.0,
) -> np.ndarray:
    """Return pixels valid for every provided depth map."""
    if not depth_maps:
        raise ValueError("At least one depth map is required")
    mask = np.ones(depth_maps[0].shape[:2], dtype=bool)
    for depth_m in depth_maps:
        mask &= valid_depth_mask(depth_m, min_depth_m=min_depth_m, max_depth_m=max_depth_m)
    return mask


def median_scale_to_reference(pred_m: np.ndarray, ref_m: np.ndarray, mask: np.ndarray) -> float:
    """Return the median scale that maps pred depth to ref depth."""
    pred = pred_m[mask]
    ref = ref_m[mask]
    if pred.size == 0:
        return float("nan")
    pred_med = float(np.median(pred))
    if abs(pred_med) < 1e-9:
        return float("nan")
    return float(np.median(ref) / pred_med)


def supervised_depth_metrics(pred_m: np.ndarray, ref_m: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Compute supervised depth errors where lower is better except delta scores."""
    pred = pred_m[mask].astype(np.float64)
    ref = ref_m[mask].astype(np.float64)
    if pred.size == 0:
        return empty_metric_dict(
            [
                "rmse",
                "mae",
                "median_abs",
                "bias",
                "abs_rel",
                "sq_rel",
                "rmse_log",
                "log10",
                "silog",
                "delta1",
                "delta2",
                "delta3",
            ]
        )

    err = pred - ref
    abs_err = np.abs(err)
    safe_pred = np.maximum(pred, 1e-6)
    safe_ref = np.maximum(ref, 1e-6)
    log_err = np.log(safe_pred) - np.log(safe_ref)
    ratio = np.maximum(safe_pred / safe_ref, safe_ref / safe_pred)
    return {
        "rmse": float(np.sqrt(np.mean(err * err))),
        "mae": float(np.mean(abs_err)),
        "median_abs": float(np.median(abs_err)),
        "bias": float(np.mean(err)),
        "abs_rel": float(np.mean(abs_err / safe_ref)),
        "sq_rel": float(np.mean((err * err) / safe_ref)),
        "rmse_log": float(np.sqrt(np.mean(log_err * log_err))),
        "log10": float(np.mean(np.abs(np.log10(safe_pred) - np.log10(safe_ref)))),
        "silog": float(np.sqrt(max(np.mean(log_err * log_err) - np.mean(log_err) ** 2, 0.0)) * 100.0),
        "delta1": float(np.mean(ratio < 1.25)),
        "delta2": float(np.mean(ratio < 1.25**2)),
        "delta3": float(np.mean(ratio < 1.25**3)),
    }


def pairwise_depth_metrics(a_m: np.ndarray, b_m: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """Compare two valid positive depth maps without an external reference."""
    a = a_m[mask].astype(np.float64)
    b = b_m[mask].astype(np.float64)
    if a.size == 0:
        return empty_metric_dict(
            [
                "mae",
                "rmse",
                "mean_rel_to_a",
                "symmetric_rel",
                "median_scale_b_over_a",
                "corr",
                "grad_mae",
            ]
        )

    diff = a - b
    abs_diff = np.abs(diff)
    safe_a = np.maximum(a, 1e-6)
    safe_b = np.maximum(b, 1e-6)
    return {
        "mae": float(np.mean(abs_diff)),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "mean_rel_to_a": float(np.mean(abs_diff / safe_a)),
        "symmetric_rel": float(np.mean(abs_diff / np.maximum((safe_a + safe_b) * 0.5, 1e-6))),
        "median_scale_b_over_a": float(np.median(b) / np.median(a)),
        "corr": float(np.corrcoef(a, b)[0, 1]) if a.size > 1 else float("nan"),
        "grad_mae": gradient_mae(a_m, b_m, mask),
    }


def gradient_mae(a_m: np.ndarray, b_m: np.ndarray, mask: np.ndarray) -> float:
    """Compare Sobel gradient magnitudes inside the valid mask."""
    import cv2

    a = np.asarray(a_m, dtype=np.float32)
    b = np.asarray(b_m, dtype=np.float32)
    grad_a_x = cv2.Sobel(a, cv2.CV_32F, 1, 0, ksize=3)
    grad_a_y = cv2.Sobel(a, cv2.CV_32F, 0, 1, ksize=3)
    grad_b_x = cv2.Sobel(b, cv2.CV_32F, 1, 0, ksize=3)
    grad_b_y = cv2.Sobel(b, cv2.CV_32F, 0, 1, ksize=3)
    mag_a = np.sqrt(grad_a_x * grad_a_x + grad_a_y * grad_a_y)
    mag_b = np.sqrt(grad_b_x * grad_b_x + grad_b_y * grad_b_y)
    values = np.abs(mag_a[mask] - mag_b[mask])
    return float(np.mean(values)) if values.size else float("nan")


def empty_metric_dict(keys: Iterable[str]) -> dict[str, float]:
    """Return a metric dict filled with NaNs."""
    return {key: float("nan") for key in keys}


def summarize_metric_rows(rows: list[dict[str, float]], keys: list[str]) -> dict[str, dict[str, float]]:
    """Summarize per-frame metric rows with mean, median, std, min, max and count."""
    summary: dict[str, dict[str, float]] = {}
    for key in keys:
        values = np.asarray([float(row[key]) for row in rows if key in row], dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            summary[key] = empty_metric_dict(["mean", "median", "std", "min", "max", "count"])
            summary[key]["count"] = 0.0
            continue
        summary[key] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "count": float(values.size),
        }
    return summary
