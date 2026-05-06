#!/usr/bin/env python3
"""Interactive viewer for ZED/DA3 GT capture samples."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


WINDOW_NAME = "ZED / DA3 interactive depth viewer"
PANEL_W = 420
RADIUS = 4


def main() -> None:
    args = parse_args()
    viewer = InteractiveDepthViewer(Path(args.path).expanduser())
    viewer.run()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="Sample dir or scene dir containing sample_* folders")
    return parser.parse_args()


class InteractiveDepthViewer:
    def __init__(self, path: Path) -> None:
        self.sample_dirs = resolve_samples(path)
        if not self.sample_dirs:
            raise FileNotFoundError(f"No sample folders found in: {path}")
        self.sample_idx = 0
        self.view_mode = "rgb"
        self.mouse_xy: tuple[int, int] | None = None
        self.sample: dict = {}
        self.load_sample(self.sample_dirs[self.sample_idx])

    def load_sample(self, sample_dir: Path) -> None:
        left = cv2.imread(str(sample_dir / "left.png"), cv2.IMREAD_COLOR)
        if left is None:
            raise FileNotFoundError(f"Missing left.png in {sample_dir}")

        self.sample = {
            "dir": sample_dir,
            "left": left,
            "zed": load_optional_depth(sample_dir / "zed_depth.npy"),
            "da3_mono": load_optional_depth(sample_dir / "da3_mono_depth.npy"),
            "da3_multiview": load_optional_depth(sample_dir / "da3_multiview_depth.npy"),
            "annotations": load_annotations(sample_dir / "annotations.csv"),
        }
        self.mouse_xy = None

    def run(self) -> None:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 1500, 850)
        cv2.setMouseCallback(WINDOW_NAME, self.on_mouse)
        try:
            while True:
                cv2.imshow(WINDOW_NAME, self.render())
                key = cv2.waitKey(30) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("v"):
                    self.next_view()
                elif key in (ord("n"), 83):
                    self.next_sample()
                elif key in (ord("p"), 81):
                    self.prev_sample()
                elif key == ord("r"):
                    self.view_mode = "rgb"
        finally:
            cv2.destroyWindow(WINDOW_NAME)

    def on_mouse(self, event: int, x: int, y: int, _flags: int, _param) -> None:
        if event not in (cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN):
            return
        h, w = self.sample["left"].shape[:2]
        if 0 <= x < w and 0 <= y < h:
            self.mouse_xy = (x, y)
        else:
            self.mouse_xy = None

    def next_view(self) -> None:
        modes = ["rgb", "zed", "da3_mono", "da3_multiview", "error_zed_da3"]
        self.view_mode = modes[(modes.index(self.view_mode) + 1) % len(modes)]

    def next_sample(self) -> None:
        self.sample_idx = min(len(self.sample_dirs) - 1, self.sample_idx + 1)
        self.load_sample(self.sample_dirs[self.sample_idx])

    def prev_sample(self) -> None:
        self.sample_idx = max(0, self.sample_idx - 1)
        self.load_sample(self.sample_dirs[self.sample_idx])

    def render(self) -> np.ndarray:
        image = self.render_main_image()
        image = draw_annotations(image, self.sample["annotations"])
        if self.mouse_xy is not None:
            x, y = self.mouse_xy
            cv2.drawMarker(image, (x, y), (0, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=18, thickness=1)

        panel = self.render_panel(image.shape[0])
        return np.hstack([image, panel])

    def render_main_image(self) -> np.ndarray:
        if self.view_mode == "rgb":
            return self.sample["left"].copy()
        if self.view_mode == "zed":
            return colorize_depth_or_empty(self.sample["zed"], self.sample["left"].shape[:2], "No ZED depth")
        if self.view_mode == "da3_mono":
            return colorize_depth_or_empty(self.sample["da3_mono"], self.sample["left"].shape[:2], "No DA3 mono depth")
        if self.view_mode == "da3_multiview":
            return colorize_depth_or_empty(self.sample["da3_multiview"], self.sample["left"].shape[:2], "No DA3 multiview depth")
        return colorize_abs_diff(self.sample["zed"], self.sample["da3_mono"], self.sample["left"].shape[:2])

    def render_panel(self, height: int) -> np.ndarray:
        panel = np.full((height, PANEL_W, 3), 28, dtype=np.uint8)
        lines = [
            f"sample: {self.sample['dir'].name} ({self.sample_idx + 1}/{len(self.sample_dirs)})",
            f"view: {self.view_mode}",
            "",
            "keys: v view | n next | p prev | r rgb | q quit",
            "",
        ]

        if self.mouse_xy is None:
            lines.append("cursor: outside image")
        else:
            x, y = self.mouse_xy
            lines.extend(self.cursor_lines(x, y))

        lines.extend(["", "annotations:"])
        for ann in self.sample["annotations"][:18]:
            name = ann.get("object_name", "")
            gt = as_float(ann.get("gt_distance_m"))
            zed = roi_median(self.sample["zed"], ann)
            da3 = roi_median(self.sample["da3_mono"], ann)
            lines.append(f"{name[:18]} GT {gt:.3f} | Z {fmt(zed)} | D {fmt(da3)}")

        y = 28
        for line in lines:
            color = (235, 235, 235) if line else (120, 120, 120)
            cv2.putText(panel, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
            y += 22
            if y > height - 12:
                break
        return panel

    def cursor_lines(self, x: int, y: int) -> list[str]:
        zed_px = depth_value(self.sample["zed"], x, y)
        da3_px = depth_value(self.sample["da3_mono"], x, y)
        multi_px = depth_value(self.sample["da3_multiview"], x, y)
        zed_med = local_median(self.sample["zed"], x, y, RADIUS)
        da3_med = local_median(self.sample["da3_mono"], x, y, RADIUS)
        multi_med = local_median(self.sample["da3_multiview"], x, y, RADIUS)
        diff = da3_med - zed_med if np.isfinite(da3_med) and np.isfinite(zed_med) else np.nan
        return [
            f"cursor: x={x} y={y}",
            f"ZED px:        {fmt(zed_px)} m",
            f"DA3 mono px:   {fmt(da3_px)} m",
            f"DA3 multi px:  {fmt(multi_px)} m",
            f"ZED median r{RADIUS}:       {fmt(zed_med)} m",
            f"DA3 mono median r{RADIUS}:  {fmt(da3_med)} m",
            f"DA3 multi median r{RADIUS}: {fmt(multi_med)} m",
            f"DA3 mono - ZED: {fmt(diff)} m",
        ]


def resolve_samples(path: Path) -> list[Path]:
    if path.name.startswith("sample_"):
        return [path]
    return sorted(p for p in path.glob("sample_*") if p.is_dir())


def load_optional_depth(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    return np.load(path)


def load_annotations(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def colorize_depth_or_empty(depth: np.ndarray | None, shape: tuple[int, int], label: str) -> np.ndarray:
    if depth is None:
        image = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
        cv2.putText(image, label, (35, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        return image
    return colorize_depth(depth)


def colorize_depth(depth: np.ndarray, min_m: float = 0.2, max_m: float | None = None) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    preview = np.zeros(depth.shape[:2], dtype=np.uint8)
    if valid.any():
        hi = float(np.percentile(depth[valid], 98)) if max_m is None else max_m
        hi = max(hi, min_m + 1e-6)
        clipped = np.clip(depth, min_m, hi)
        norm = 1.0 - (clipped - min_m) / (hi - min_m)
        preview[valid] = (norm[valid] * 255.0).astype(np.uint8)
    return cv2.applyColorMap(preview, cv2.COLORMAP_TURBO)


def colorize_abs_diff(a: np.ndarray | None, b: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray:
    if a is None or b is None:
        image = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
        cv2.putText(image, "Need ZED and DA3 mono depth", (35, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        return image
    diff = np.abs(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32))
    valid = np.isfinite(diff)
    preview = np.zeros(diff.shape[:2], dtype=np.uint8)
    if valid.any():
        clipped = np.clip(diff, 0.0, 1.0)
        preview[valid] = (clipped[valid] * 255.0).astype(np.uint8)
    return cv2.applyColorMap(preview, cv2.COLORMAP_INFERNO)


def draw_annotations(image: np.ndarray, annotations: list[dict[str, str]]) -> np.ndarray:
    out = image.copy()
    for ann in annotations:
        x1, y1, x2, y2 = [int(float(ann[key])) for key in ("x1", "y1", "x2", "y2")]
        cv2.rectangle(out, (x1, y1), (x2, y2), (20, 230, 80), 2)
        label = f"{ann.get('object_name', '')} GT {as_float(ann.get('gt_distance_m')):.2f}m"
        cv2.putText(out, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 230, 80), 1, cv2.LINE_AA)
    return out


def depth_value(depth: np.ndarray | None, x: int, y: int) -> float:
    if depth is None:
        return np.nan
    if y < 0 or x < 0 or y >= depth.shape[0] or x >= depth.shape[1]:
        return np.nan
    value = float(depth[y, x])
    return value if np.isfinite(value) and value > 0 else np.nan


def local_median(depth: np.ndarray | None, x: int, y: int, radius: int) -> float:
    if depth is None:
        return np.nan
    x1 = max(0, x - radius)
    x2 = min(depth.shape[1], x + radius + 1)
    y1 = max(0, y - radius)
    y2 = min(depth.shape[0], y + radius + 1)
    crop = depth[y1:y2, x1:x2]
    valid = crop[np.isfinite(crop) & (crop > 0)]
    if valid.size == 0:
        return np.nan
    return float(np.median(valid))


def roi_median(depth: np.ndarray | None, ann: dict[str, str]) -> float:
    if depth is None:
        return np.nan
    x1, y1, x2, y2 = [int(float(ann[key])) for key in ("x1", "y1", "x2", "y2")]
    crop = depth[y1:y2, x1:x2]
    valid = crop[np.isfinite(crop) & (crop > 0)]
    return float(np.median(valid)) if valid.size else np.nan


def as_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def fmt(value: float) -> str:
    return f"{value:.3f}" if np.isfinite(value) else "nan"


if __name__ == "__main__":
    main()
