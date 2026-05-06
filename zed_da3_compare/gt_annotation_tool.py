#!/usr/bin/env python3
"""Live/offline ROI annotation tool for ZED vs DA3 ground-truth comparisons."""

from __future__ import annotations

import csv
import json
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Deque

import cv2
import numpy as np
import rclpy
import torch
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import Image

from zed_da3_compare.depth_metrics import resize_depth_to_shape
from zed_da3_compare.ros_image_utils import depth_msg_to_meters, image_msg_to_rgb


DEFAULT_CAPTURE_ROOT = Path("/home/usuario/depth_anything_ws/src/zed_da3_compare/captures")
WINDOW_NAME = "ZED / DA3 GT Annotation Tool"
TOOLBAR_H = 86
ANNOTATION_FIELDS = [
    "scene",
    "sample_id",
    "object_id",
    "object_name",
    "gt_distance_m",
    "x1",
    "y1",
    "x2",
    "y2",
    "zed_median_m",
    "zed_mean_m",
    "da3_mono_median_m",
    "da3_mono_mean_m",
    "da3_multiview_median_m",
    "da3_multiview_mean_m",
    "zed_abs_error_m",
    "da3_mono_abs_error_m",
    "da3_multiview_abs_error_m",
    "winner_abs_error",
    "notes",
]


@dataclass
class LiveRecord:
    stamp_ns: int
    msg: Image


@dataclass
class SyncedLiveTriple:
    left: LiveRecord
    right: LiveRecord | None
    depth: LiveRecord

    @property
    def left_depth_delta_ms(self) -> float:
        return abs(self.left.stamp_ns - self.depth.stamp_ns) / 1e6

    @property
    def left_right_delta_ms(self) -> float:
        if self.right is None:
            return float("nan")
        return abs(self.left.stamp_ns - self.right.stamp_ns) / 1e6


@dataclass
class Button:
    label: str
    rect: tuple[int, int, int, int]
    action: Callable[[], None]


def stamp_ns(msg: Image) -> int:
    return int(msg.header.stamp.sec) * 1_000_000_000 + int(msg.header.stamp.nanosec)


def nearest_record(reference: LiveRecord, records: Deque[LiveRecord]) -> tuple[LiveRecord | None, float]:
    if not records:
        return None, float("inf")
    best = min(records, key=lambda record: abs(record.stamp_ns - reference.stamp_ns))
    return best, abs(best.stamp_ns - reference.stamp_ns) / 1e6


def prompt_text(label: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    value = input(f"{label}{suffix}: ").strip()
    return value if value else default


def prompt_float(label: str, default: float | None = None) -> float:
    while True:
        suffix = f" [{default}]" if default is not None else ""
        value = input(f"{label}{suffix}: ").strip()
        if not value and default is not None:
            return float(default)
        try:
            return float(value)
        except ValueError:
            print("Introduce un numero en metros, por ejemplo 1.35")


def colorize_depth(depth_m: np.ndarray, min_m: float = 0.2, max_m: float = 8.0) -> np.ndarray:
    depth_m = np.asarray(depth_m, dtype=np.float32)
    valid = np.isfinite(depth_m) & (depth_m > 0.0)
    preview = np.zeros(depth_m.shape, dtype=np.uint8)
    if valid.any():
        clipped = np.clip(depth_m, min_m, max_m)
        norm = 1.0 - ((clipped - min_m) / max(max_m - min_m, 1e-6))
        preview[valid] = (norm[valid] * 255.0).astype(np.uint8)
    return cv2.applyColorMap(preview, cv2.COLORMAP_TURBO)


def roi_stats(depth_m: np.ndarray | None, roi: tuple[int, int, int, int]) -> tuple[float, float]:
    if depth_m is None:
        return float("nan"), float("nan")
    x1, y1, x2, y2 = roi
    crop = depth_m[y1:y2, x1:x2]
    valid = crop[np.isfinite(crop) & (crop > 0.0)]
    if valid.size == 0:
        return float("nan"), float("nan")
    return float(np.median(valid)), float(np.mean(valid))


def abs_error(value: float, gt: float) -> float:
    return abs(value - gt) if np.isfinite(value) else float("nan")


def winner_from_errors(errors: dict[str, float]) -> str:
    finite = {name: value for name, value in errors.items() if np.isfinite(value)}
    if not finite:
        return "none"
    return min(finite, key=finite.get)


class GtAnnotationTool(Node):
    def __init__(self) -> None:
        super().__init__("gt_annotation_tool")
        self.declare_parameter("captures_root", str(DEFAULT_CAPTURE_ROOT))
        self.declare_parameter("scene_name", "scene_01")
        self.declare_parameter("left_image_topic", "/zed/zed_node/rgb/color/rect/image")
        self.declare_parameter("right_image_topic", "/zed/zed_node/right/color/rect/image")
        self.declare_parameter("zed_depth_topic", "/zed/zed_node/depth/depth_registered")
        self.declare_parameter("sync_tolerance_ms", 30.0)
        self.declare_parameter("queue_size", 120)
        self.declare_parameter("model_dir", os.environ.get("DA3_MODEL_DIR", ""))
        self.declare_parameter("device", "cuda")
        self.declare_parameter("process_res", 504)
        self.declare_parameter("process_res_method", "upper_bound_resize")
        self.declare_parameter("ref_view_strategy", "saddle_balanced")
        self.declare_parameter("process_da3_on_capture", False)

        queue_size = max(3, int(self.get_parameter("queue_size").value))
        self.left_queue: Deque[LiveRecord] = deque(maxlen=queue_size)
        self.right_queue: Deque[LiveRecord] = deque(maxlen=queue_size)
        self.depth_queue: Deque[LiveRecord] = deque(maxlen=queue_size)
        self.captures_root = Path(str(self.get_parameter("captures_root").value)).expanduser()
        self.scene_name = str(self.get_parameter("scene_name").value)
        self.process_da3_on_capture = bool(self.get_parameter("process_da3_on_capture").value)
        self.model = None

        self.mode = "live"
        self.view = "rgb"
        self.status = "Ready. Click Settings to set scene, Capture to freeze a synced sample."
        self.current_sample: dict[str, Any] | None = None
        self.current_sample_dir: Path | None = None
        self.sample_dirs: list[Path] = []
        self.sample_index = -1
        self.annotations: list[dict[str, Any]] = []

        self.buttons: list[Button] = []
        self.dragging_roi = False
        self.roi_start: tuple[int, int] | None = None
        self.roi_current: tuple[int, int] | None = None
        self.display_scale = 1.0
        self.display_origin = (0, TOOLBAR_H)

        self.create_subscription(Image, str(self.get_parameter("left_image_topic").value), self.on_left, 10)
        self.create_subscription(Image, str(self.get_parameter("right_image_topic").value), self.on_right, 10)
        self.create_subscription(Image, str(self.get_parameter("zed_depth_topic").value), self.on_depth, 10)

        self.get_logger().info("GT annotation tool started")
        self.get_logger().info("Use GUI buttons or shortcuts: c capture, a add ROI, p process DA3, s save, q quit")

    def on_left(self, msg: Image) -> None:
        self.left_queue.append(LiveRecord(stamp_ns(msg), msg))

    def on_right(self, msg: Image) -> None:
        self.right_queue.append(LiveRecord(stamp_ns(msg), msg))

    def on_depth(self, msg: Image) -> None:
        self.depth_queue.append(LiveRecord(stamp_ns(msg), msg))

    def find_synced_triple(self) -> SyncedLiveTriple | None:
        tolerance_ms = float(self.get_parameter("sync_tolerance_ms").value)
        for left in reversed(self.left_queue):
            depth, depth_delta = nearest_record(left, self.depth_queue)
            if depth is None or depth_delta > tolerance_ms:
                continue
            right, right_delta = nearest_record(left, self.right_queue)
            if right is not None and right_delta > tolerance_ms:
                right = None
            return SyncedLiveTriple(left=left, right=right, depth=depth)
        return None

    def sync_debug_text(self) -> str:
        tolerance_ms = float(self.get_parameter("sync_tolerance_ms").value)
        if not self.left_queue:
            return f"No RGB/left messages yet. queues left=0 right={len(self.right_queue)} depth={len(self.depth_queue)}"
        latest_left = self.left_queue[-1]
        depth, depth_delta = nearest_record(latest_left, self.depth_queue)
        right, right_delta = nearest_record(latest_left, self.right_queue)
        depth_text = f"{depth_delta:.1f}ms" if depth is not None else "none"
        right_text = f"{right_delta:.1f}ms" if right is not None else "none"
        return (
            f"No synced RGB/depth <= {tolerance_ms:.1f}ms. "
            f"queues left={len(self.left_queue)} right={len(self.right_queue)} depth={len(self.depth_queue)} "
            f"nearest depth={depth_text} right={right_text}. "
            "Try rgb/color/rect/image or increase sync_tolerance_ms."
        )

    def run(self) -> None:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 1280, 820)
        cv2.setMouseCallback(WINDOW_NAME, self.on_mouse)
        try:
            while rclpy.ok():
                rclpy.spin_once(self, timeout_sec=0.01)
                frame = self.render()
                cv2.imshow(WINDOW_NAME, frame)
                key = cv2.waitKey(30) & 0xFF
                if self.handle_key(key):
                    break
        finally:
            cv2.destroyWindow(WINDOW_NAME)

    def handle_key(self, key: int) -> bool:
        if key in (ord("q"), 27):
            return True
        if key == ord("c"):
            self.capture_live_sample()
        elif key == ord("a"):
            self.start_roi_mode()
        elif key == ord("s"):
            self.save_current_sample()
        elif key == ord("p"):
            self.process_current_da3()
        elif key == ord("e"):
            self.edit_annotation()
        elif key == ord("v"):
            self.toggle_view()
        elif key == ord("o"):
            self.open_scene_or_sample()
        elif key == ord("l"):
            self.mode = "live"
            self.status = "Live mode"
        return False

    def on_mouse(self, event: int, x: int, y: int, _flags: int, _param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            for button in self.buttons:
                x1, y1, x2, y2 = button.rect
                if x1 <= x <= x2 and y1 <= y <= y2:
                    button.action()
                    return
            image_point = self.screen_to_image_point(x, y)
            if self.current_sample is not None and image_point is not None:
                self.dragging_roi = True
                self.roi_start = image_point
                self.roi_current = image_point
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging_roi:
            image_point = self.screen_to_image_point(x, y)
            if image_point is not None:
                self.roi_current = image_point
        elif event == cv2.EVENT_LBUTTONUP and self.dragging_roi:
            self.dragging_roi = False
            if self.roi_start is None:
                return
            end = self.screen_to_image_point(x, y)
            if end is None:
                return
            self.add_annotation_from_roi(self.normalize_roi(self.roi_start, end))
            self.roi_start = None
            self.roi_current = None

    def screen_to_image_point(self, x: int, y: int) -> tuple[int, int] | None:
        if self.current_sample is None:
            return None
        origin_x, origin_y = self.display_origin
        image = self.display_image_for_current_sample()
        img_x = int((x - origin_x) / max(self.display_scale, 1e-6))
        img_y = int((y - origin_y) / max(self.display_scale, 1e-6))
        if img_x < 0 or img_y < 0 or img_x >= image.shape[1] or img_y >= image.shape[0]:
            return None
        return img_x, img_y

    def normalize_roi(self, start: tuple[int, int], end: tuple[int, int]) -> tuple[int, int, int, int]:
        image = self.display_image_for_current_sample()
        h, w = image.shape[:2]
        x1, x2 = sorted([max(0, min(w - 1, start[0])), max(0, min(w - 1, end[0]))])
        y1, y2 = sorted([max(0, min(h - 1, start[1])), max(0, min(h - 1, end[1]))])
        return x1, y1, max(x1 + 1, x2), max(y1 + 1, y2)

    def render(self) -> np.ndarray:
        canvas = np.zeros((820, 1280, 3), dtype=np.uint8)
        self.draw_toolbar(canvas)
        image = self.display_image()
        max_h, max_w = canvas.shape[0] - TOOLBAR_H, canvas.shape[1]
        scale = min(max_w / image.shape[1], max_h / image.shape[0])
        self.display_scale = scale
        self.display_origin = (0, TOOLBAR_H)
        resized = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))
        canvas[TOOLBAR_H : TOOLBAR_H + resized.shape[0], : resized.shape[1]] = resized
        self.draw_annotations(canvas, scale)
        return canvas

    def draw_toolbar(self, canvas: np.ndarray) -> None:
        canvas[:TOOLBAR_H, :] = (32, 32, 32)
        labels_actions = [
            ("Settings", self.configure),
            ("Capture", self.capture_live_sample),
            ("Add ROI", self.start_roi_mode),
            ("Process DA3", self.process_current_da3),
            ("Save", self.save_current_sample),
            ("Open", self.open_scene_or_sample),
            ("Bag", self.import_bag_as_samples),
            ("Edit", self.edit_annotation),
            ("Prev", self.open_prev_sample),
            ("Next", self.open_next_sample),
            ("View", self.toggle_view),
            ("Live", self.go_live),
            ("Quit", self.request_quit),
        ]
        self.buttons = []
        x = 8
        for label, action in labels_actions:
            if label == "Process DA3":
                w = 108
            elif label in {"Settings", "Capture", "Add ROI"}:
                w = 78
            else:
                w = 60
            rect = (x, 8, x + w, 38)
            self.buttons.append(Button(label=label, rect=rect, action=action))
            cv2.rectangle(canvas, rect[:2], rect[2:], (78, 114, 160), -1)
            cv2.putText(canvas, label, (x + 6, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
            x += w + 8
        cv2.putText(
            canvas,
            f"scene={self.scene_name} mode={self.mode} view={self.view} samples={len(self.sample_dirs)} "
            f"process_on_capture={self.process_da3_on_capture}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (230, 230, 230),
            1,
        )
        cv2.putText(canvas, self.status[:150], (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (170, 230, 170), 1)

    def display_image(self) -> np.ndarray:
        if self.current_sample is not None:
            return self.display_image_for_current_sample()
        latest = self.left_queue[-1].msg if self.left_queue else None
        if latest is None:
            image = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(image, "Waiting for live image topic...", (40, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            return image
        return cv2.cvtColor(image_msg_to_rgb(latest), cv2.COLOR_RGB2BGR)

    def display_image_for_current_sample(self) -> np.ndarray:
        assert self.current_sample is not None
        if self.view == "zed" and self.current_sample.get("zed_depth") is not None:
            return colorize_depth(self.current_sample["zed_depth"])
        if self.view == "da3_mono" and self.current_sample.get("da3_mono_depth") is not None:
            return colorize_depth(self.current_sample["da3_mono_depth"])
        if self.view == "da3_multiview" and self.current_sample.get("da3_multiview_depth") is not None:
            return colorize_depth(self.current_sample["da3_multiview_depth"])
        return cv2.cvtColor(self.current_sample["left_rgb"], cv2.COLOR_RGB2BGR)

    def sample_verification_metadata(
        self,
        triple: SyncedLiveTriple,
        left_rgb: np.ndarray,
        right_rgb: np.ndarray | None,
        zed_depth: np.ndarray,
    ) -> dict[str, Any]:
        valid = zed_depth[np.isfinite(zed_depth) & (zed_depth > 0.0)]
        depth_stats = {
            "valid_ratio": float(valid.size / max(zed_depth.size, 1)),
            "min_m": float(np.min(valid)) if valid.size else None,
            "median_m": float(np.median(valid)) if valid.size else None,
            "mean_m": float(np.mean(valid)) if valid.size else None,
            "max_m": float(np.max(valid)) if valid.size else None,
        }
        return {
            "left_encoding": triple.left.msg.encoding,
            "right_encoding": triple.right.msg.encoding if triple.right else None,
            "depth_encoding": triple.depth.msg.encoding,
            "left_shape_hw": list(left_rgb.shape[:2]),
            "right_shape_hw": list(right_rgb.shape[:2]) if right_rgb is not None else None,
            "depth_shape_hw": list(zed_depth.shape[:2]),
            "rgb_depth_same_shape": list(left_rgb.shape[:2]) == list(zed_depth.shape[:2]),
            "right_available": right_rgb is not None,
            "zed_depth_stats": depth_stats,
        }

    def draw_annotations(self, canvas: np.ndarray, scale: float) -> None:
        if self.current_sample is None:
            return
        origin_x, origin_y = self.display_origin
        for ann in self.annotations:
            x1, y1, x2, y2 = [int(float(ann[key]) * scale) for key in ("x1", "y1", "x2", "y2")]
            x1 += origin_x
            x2 += origin_x
            y1 += origin_y
            y2 += origin_y
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (30, 220, 80), 2)
            label = f"{ann.get('object_name', '')} GT={ann.get('gt_distance_m', '')}m"
            cv2.putText(canvas, label, (x1, max(origin_y + 18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (30, 240, 90), 2)
        if self.dragging_roi and self.roi_start and self.roi_current:
            x1, y1, x2, y2 = self.normalize_roi(self.roi_start, self.roi_current)
            cv2.rectangle(
                canvas,
                (origin_x + int(x1 * scale), origin_y + int(y1 * scale)),
                (origin_x + int(x2 * scale), origin_y + int(y2 * scale)),
                (0, 220, 255),
                2,
            )

    def configure(self) -> None:
        self.scene_name = prompt_text("Scene name", self.scene_name)
        process = prompt_text("Process DA3 on capture? y/n", "y" if self.process_da3_on_capture else "n").lower()
        self.process_da3_on_capture = process.startswith("y")
        model_dir = prompt_text("DA3 model dir (empty keeps current/env)", str(self.get_parameter("model_dir").value))
        self.set_parameters([Parameter("model_dir", Parameter.Type.STRING, model_dir)])
        self.status = "Settings updated"

    def capture_live_sample(self) -> None:
        triple = self.find_synced_triple()
        if triple is None:
            self.status = self.sync_debug_text()
            return
        left_rgb = image_msg_to_rgb(triple.left.msg)
        right_rgb = image_msg_to_rgb(triple.right.msg) if triple.right is not None else None
        zed_depth = depth_msg_to_meters(triple.depth.msg)
        verification = self.sample_verification_metadata(triple, left_rgb, right_rgb, zed_depth)
        self.current_sample = {
            "left_rgb": left_rgb,
            "right_rgb": right_rgb,
            "zed_depth": zed_depth,
            "da3_mono_depth": None,
            "da3_multiview_depth": None,
            "metadata": {
                "scene": self.scene_name,
                "source": "live",
                "left_stamp_ns": triple.left.stamp_ns,
                "right_stamp_ns": triple.right.stamp_ns if triple.right else None,
                "depth_stamp_ns": triple.depth.stamp_ns,
                "left_depth_delta_ms": triple.left_depth_delta_ms,
                "left_right_delta_ms": triple.left_right_delta_ms,
                "left_topic": str(self.get_parameter("left_image_topic").value),
                "right_topic": str(self.get_parameter("right_image_topic").value),
                "zed_depth_topic": str(self.get_parameter("zed_depth_topic").value),
                "verification": verification,
            },
        }
        self.annotations = []
        self.current_sample_dir = self.next_sample_dir()
        self.mode = "sample"
        self.save_current_sample()
        if self.process_da3_on_capture:
            self.process_current_da3()
        self.status = f"Captured {self.current_sample_dir.name}. Drag on image to add ROI."

    def next_sample_dir(self) -> Path:
        scene_dir = self.captures_root / self.scene_name
        scene_dir.mkdir(parents=True, exist_ok=True)
        idx = 1
        while True:
            candidate = scene_dir / f"sample_{idx:06d}"
            if not candidate.exists():
                return candidate
            idx += 1

    def start_roi_mode(self) -> None:
        if self.current_sample is None:
            self.status = "Capture or open a sample before adding ROIs"
            return
        self.status = "Drag a rectangle over the object. GT distance is in metres."

    def add_annotation_from_roi(self, roi: tuple[int, int, int, int]) -> None:
        if self.current_sample is None:
            return
        if (roi[2] - roi[0]) < 4 or (roi[3] - roi[1]) < 4:
            self.status = "ROI too small"
            return
        object_name = prompt_text("Object name", f"object_{len(self.annotations) + 1:02d}")
        gt_distance = prompt_float("GT distance in metres")
        notes = prompt_text("Notes", "")
        ann = self.build_annotation(roi, object_name, gt_distance, notes)
        self.annotations.append(ann)
        self.save_current_sample()
        self.status = f"Added ROI {ann['object_id']} {object_name} GT={gt_distance:.3f}m"

    def edit_annotation(self) -> None:
        if not self.annotations:
            self.status = "No annotations to edit"
            return
        print("\nAnnotations:")
        for idx, ann in enumerate(self.annotations, start=1):
            print(
                f"{idx}. {ann.get('object_id')} {ann.get('object_name')} "
                f"GT={ann.get('gt_distance_m')} ROI=({ann.get('x1')},{ann.get('y1')},{ann.get('x2')},{ann.get('y2')})"
            )
        index = int(prompt_float("Annotation number to edit", 1.0)) - 1
        if index < 0 or index >= len(self.annotations):
            self.status = "Invalid annotation number"
            return
        old = self.annotations[index]
        object_name = prompt_text("Object name", str(old.get("object_name", "")))
        gt_distance = prompt_float("GT distance in metres", float(old.get("gt_distance_m", 0.0)))
        x1 = int(prompt_float("x1", float(old.get("x1", 0))))
        y1 = int(prompt_float("y1", float(old.get("y1", 0))))
        x2 = int(prompt_float("x2", float(old.get("x2", 1))))
        y2 = int(prompt_float("y2", float(old.get("y2", 1))))
        notes = prompt_text("Notes", str(old.get("notes", "")))
        new_ann = self.build_annotation(self.normalize_roi((x1, y1), (x2, y2)), object_name, gt_distance, notes)
        new_ann["object_id"] = old.get("object_id", new_ann["object_id"])
        self.annotations[index] = new_ann
        self.save_current_sample()
        self.status = f"Edited {new_ann['object_id']}"

    def build_annotation(self, roi: tuple[int, int, int, int], object_name: str, gt_distance: float, notes: str) -> dict[str, Any]:
        assert self.current_sample is not None
        zed_med, zed_mean = roi_stats(self.current_sample.get("zed_depth"), roi)
        mono_med, mono_mean = roi_stats(self.current_sample.get("da3_mono_depth"), roi)
        multi_med, multi_mean = roi_stats(self.current_sample.get("da3_multiview_depth"), roi)
        errors = {
            "zed": abs_error(zed_med, gt_distance),
            "da3_mono": abs_error(mono_med, gt_distance),
            "da3_multiview": abs_error(multi_med, gt_distance),
        }
        sample_id = self.current_sample_dir.name if self.current_sample_dir else "unsaved"
        return {
            "scene": self.scene_name,
            "sample_id": sample_id,
            "object_id": f"obj_{len(self.annotations) + 1:03d}",
            "object_name": object_name,
            "gt_distance_m": gt_distance,
            "x1": roi[0],
            "y1": roi[1],
            "x2": roi[2],
            "y2": roi[3],
            "zed_median_m": zed_med,
            "zed_mean_m": zed_mean,
            "da3_mono_median_m": mono_med,
            "da3_mono_mean_m": mono_mean,
            "da3_multiview_median_m": multi_med,
            "da3_multiview_mean_m": multi_mean,
            "zed_abs_error_m": errors["zed"],
            "da3_mono_abs_error_m": errors["da3_mono"],
            "da3_multiview_abs_error_m": errors["da3_multiview"],
            "winner_abs_error": winner_from_errors(errors),
            "notes": notes,
        }

    def process_current_da3(self) -> None:
        if self.current_sample is None:
            self.status = "No current sample to process"
            return
        model_dir = str(self.get_parameter("model_dir").value)
        if not model_dir:
            self.status = "Set DA3 model_dir in Settings or DA3_MODEL_DIR"
            return
        if self.model is None:
            from depth_anything_3.api import DepthAnything3

            self.status = "Loading DA3 model..."
            self.model = DepthAnything3.from_pretrained(model_dir).to(str(self.get_parameter("device").value)).eval()
        left_rgb = self.current_sample["left_rgb"]
        right_rgb = self.current_sample.get("right_rgb")
        process_res = int(self.get_parameter("process_res").value)
        process_res_method = str(self.get_parameter("process_res_method").value)
        with torch.inference_mode():
            mono_prediction = self.model.inference(
                image=[left_rgb],
                process_res=process_res,
                process_res_method=process_res_method,
                export_dir=None,
            )
            mono = np.asarray(mono_prediction.depth[0], dtype=np.float32)
            mono = resize_depth_to_shape(mono, left_rgb.shape[:2])
            self.current_sample["da3_mono_depth"] = mono
            if right_rgb is not None:
                multi_prediction = self.model.inference(
                    image=[left_rgb, right_rgb],
                    process_res=process_res,
                    process_res_method=process_res_method,
                    ref_view_strategy=str(self.get_parameter("ref_view_strategy").value),
                    export_dir=None,
                )
                multi = np.asarray(multi_prediction.depth[0], dtype=np.float32)
                self.current_sample["da3_multiview_depth"] = resize_depth_to_shape(multi, left_rgb.shape[:2])
        self.recompute_annotations()
        self.save_current_sample()
        self.status = "DA3 processed for current sample"

    def recompute_annotations(self) -> None:
        old_annotations = list(self.annotations)
        self.annotations = []
        for old in old_annotations:
            roi = tuple(int(old[key]) for key in ("x1", "y1", "x2", "y2"))
            self.annotations.append(
                self.build_annotation(roi, str(old["object_name"]), float(old["gt_distance_m"]), str(old.get("notes", "")))
            )

    def save_current_sample(self) -> None:
        if self.current_sample is None:
            self.status = "No current sample to save"
            return
        if self.current_sample_dir is None:
            self.current_sample_dir = self.next_sample_dir()
        self.current_sample_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(self.current_sample_dir / "left.png"), cv2.cvtColor(self.current_sample["left_rgb"], cv2.COLOR_RGB2BGR))
        if self.current_sample.get("right_rgb") is not None:
            cv2.imwrite(str(self.current_sample_dir / "right.png"), cv2.cvtColor(self.current_sample["right_rgb"], cv2.COLOR_RGB2BGR))
        np.save(self.current_sample_dir / "zed_depth.npy", self.current_sample["zed_depth"])
        if self.current_sample.get("da3_mono_depth") is not None:
            np.save(self.current_sample_dir / "da3_mono_depth.npy", self.current_sample["da3_mono_depth"])
        if self.current_sample.get("da3_multiview_depth") is not None:
            np.save(self.current_sample_dir / "da3_multiview_depth.npy", self.current_sample["da3_multiview_depth"])
        metadata = dict(self.current_sample.get("metadata", {}))
        metadata.update(
            {
                "scene": self.scene_name,
                "sample_id": self.current_sample_dir.name,
                "has_da3_mono": self.current_sample.get("da3_mono_depth") is not None,
                "has_da3_multiview": self.current_sample.get("da3_multiview_depth") is not None,
                "annotation_count": len(self.annotations),
            }
        )
        (self.current_sample_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        self.write_annotations_csv(self.current_sample_dir / "annotations.csv", self.annotations)
        self.write_scene_annotations()
        self.write_roi_preview()
        self.refresh_sample_dirs()
        self.status = f"Saved {self.current_sample_dir}"

    def write_annotations_csv(self, path: Path, rows: list[dict[str, Any]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=ANNOTATION_FIELDS)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: row.get(field, "") for field in ANNOTATION_FIELDS})

    def write_scene_annotations(self) -> None:
        scene_dir = self.captures_root / self.scene_name
        rows: list[dict[str, Any]] = []
        for sample_dir in sorted(scene_dir.glob("sample_*")):
            csv_path = sample_dir / "annotations.csv"
            if not csv_path.exists():
                continue
            with csv_path.open("r", newline="", encoding="utf-8") as handle:
                rows.extend(csv.DictReader(handle))
        self.write_annotations_csv(scene_dir / "scene_annotations.csv", rows)

    def write_roi_preview(self) -> None:
        if self.current_sample is None or self.current_sample_dir is None:
            return
        preview = cv2.cvtColor(self.current_sample["left_rgb"], cv2.COLOR_RGB2BGR)
        for ann in self.annotations:
            x1, y1, x2, y2 = [int(float(ann[key])) for key in ("x1", "y1", "x2", "y2")]
            cv2.rectangle(preview, (x1, y1), (x2, y2), (30, 220, 80), 2)
            cv2.putText(preview, f"{ann['object_name']} {ann['gt_distance_m']}m", (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 220, 80), 2)
        cv2.imwrite(str(self.current_sample_dir / "preview_rois.png"), preview)

    def open_scene_or_sample(self) -> None:
        path_text = prompt_text("Scene dir or sample dir", str(self.captures_root / self.scene_name))
        path = Path(path_text).expanduser()
        if not path.exists():
            self.status = f"Path does not exist: {path}"
            return
        if path.name.startswith("sample_"):
            self.open_sample(path)
            self.sample_dirs = sorted(path.parent.glob("sample_*"))
            self.sample_index = self.sample_dirs.index(path) if path in self.sample_dirs else -1
        else:
            self.sample_dirs = sorted(path.glob("sample_*"))
            self.sample_index = 0 if self.sample_dirs else -1
            if self.sample_index >= 0:
                self.open_sample(self.sample_dirs[self.sample_index])
        self.mode = "offline"

    def open_sample(self, sample_dir: Path) -> None:
        left = cv2.imread(str(sample_dir / "left.png"), cv2.IMREAD_COLOR)
        if left is None:
            self.status = f"Could not read {sample_dir / 'left.png'}"
            return
        right_path = sample_dir / "right.png"
        right = cv2.imread(str(right_path), cv2.IMREAD_COLOR) if right_path.exists() else None
        metadata_path = sample_dir / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        self.scene_name = str(metadata.get("scene", sample_dir.parent.name))
        self.current_sample_dir = sample_dir
        self.current_sample = {
            "left_rgb": cv2.cvtColor(left, cv2.COLOR_BGR2RGB),
            "right_rgb": cv2.cvtColor(right, cv2.COLOR_BGR2RGB) if right is not None else None,
            "zed_depth": np.load(sample_dir / "zed_depth.npy") if (sample_dir / "zed_depth.npy").exists() else None,
            "da3_mono_depth": np.load(sample_dir / "da3_mono_depth.npy") if (sample_dir / "da3_mono_depth.npy").exists() else None,
            "da3_multiview_depth": np.load(sample_dir / "da3_multiview_depth.npy") if (sample_dir / "da3_multiview_depth.npy").exists() else None,
            "metadata": metadata,
        }
        self.annotations = self.read_annotations(sample_dir / "annotations.csv")
        self.status = f"Opened {sample_dir}"

    def import_bag_as_samples(self) -> None:
        bag = prompt_text("Bag path or name inside zed_da3_compare/data")
        if not bag:
            self.status = "Bag import cancelled"
            return
        scene = prompt_text("Scene name for imported samples", self.scene_name)
        left_topic = prompt_text("Bag left image topic", str(self.get_parameter("left_image_topic").value))
        right_topic = prompt_text("Bag right image topic", str(self.get_parameter("right_image_topic").value))
        depth_topic = prompt_text("Bag ZED depth topic", str(self.get_parameter("zed_depth_topic").value))
        max_frames = int(prompt_float("Max frames to import, 0 means all", 20.0))
        from zed_da3_compare.da3_offline_bag_eval import sync_bag_triples

        args = SimpleNamespace(
            bag=bag,
            left_topic=left_topic,
            right_topic=right_topic,
            zed_depth_topic=depth_topic,
            sync_tolerance_ms=float(self.get_parameter("sync_tolerance_ms").value),
            zed_depth_time_offset_ms=0.0,
            storage_id="",
        )
        previous_scene = self.scene_name
        self.scene_name = scene
        imported = 0
        try:
            for triple in sync_bag_triples(args):
                left_rgb = image_msg_to_rgb(triple.left.msg)
                right_rgb = image_msg_to_rgb(triple.right.msg)
                zed_depth = depth_msg_to_meters(triple.zed_depth.msg)
                self.current_sample = {
                    "left_rgb": left_rgb,
                    "right_rgb": right_rgb,
                    "zed_depth": zed_depth,
                    "da3_mono_depth": None,
                    "da3_multiview_depth": None,
                    "metadata": {
                        "scene": self.scene_name,
                        "source": "bag",
                        "bag": bag,
                        "source_idx": triple.index,
                        "left_stamp_ns": triple.left.stamp_ns,
                        "right_stamp_ns": triple.right.stamp_ns,
                        "depth_stamp_ns": triple.zed_depth.stamp_ns,
                        "left_depth_delta_ms": triple.left_depth_delta_ms,
                        "left_right_delta_ms": triple.left_right_delta_ms,
                        "left_topic": args.left_topic,
                        "right_topic": args.right_topic,
                        "zed_depth_topic": args.zed_depth_topic,
                    },
                }
                self.annotations = []
                self.current_sample_dir = self.next_sample_dir()
                self.save_current_sample()
                imported += 1
                if max_frames > 0 and imported >= max_frames:
                    break
        except Exception as exc:
            self.scene_name = previous_scene
            self.status = f"Bag import failed: {exc}"
            return
        self.refresh_sample_dirs()
        if self.sample_dirs:
            self.sample_index = len(self.sample_dirs) - 1
            self.open_sample(self.sample_dirs[self.sample_index])
        self.mode = "offline"
        self.status = f"Imported {imported} samples from bag into scene {self.scene_name}"

    def read_annotations(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        with path.open("r", newline="", encoding="utf-8") as handle:
            return [dict(row) for row in csv.DictReader(handle)]

    def refresh_sample_dirs(self) -> None:
        scene_dir = self.captures_root / self.scene_name
        self.sample_dirs = sorted(scene_dir.glob("sample_*")) if scene_dir.exists() else []
        if self.current_sample_dir in self.sample_dirs:
            self.sample_index = self.sample_dirs.index(self.current_sample_dir)

    def open_prev_sample(self) -> None:
        if not self.sample_dirs:
            self.refresh_sample_dirs()
        if not self.sample_dirs:
            self.status = "No samples in current scene"
            return
        self.sample_index = max(0, self.sample_index - 1)
        self.open_sample(self.sample_dirs[self.sample_index])

    def open_next_sample(self) -> None:
        if not self.sample_dirs:
            self.refresh_sample_dirs()
        if not self.sample_dirs:
            self.status = "No samples in current scene"
            return
        self.sample_index = min(len(self.sample_dirs) - 1, self.sample_index + 1)
        self.open_sample(self.sample_dirs[self.sample_index])

    def toggle_view(self) -> None:
        views = ["rgb", "zed", "da3_mono", "da3_multiview"]
        self.view = views[(views.index(self.view) + 1) % len(views)]
        self.status = f"View: {self.view}"

    def go_live(self) -> None:
        self.mode = "live"
        self.current_sample = None
        self.current_sample_dir = None
        self.annotations = []
        self.status = "Live mode"

    def request_quit(self) -> None:
        raise KeyboardInterrupt


def main() -> None:
    rclpy.init()
    node = GtAnnotationTool()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
