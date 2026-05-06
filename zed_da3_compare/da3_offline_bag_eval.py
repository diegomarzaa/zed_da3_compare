#!/usr/bin/env python3
"""Offline DA3 mono/multiview evaluation from ZED ROS 2 bags."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from zed_da3_compare.depth_metrics import (
    joint_valid_mask,
    median_scale_to_reference,
    pairwise_depth_metrics,
    resize_depth_to_shape,
    summarize_metric_rows,
    supervised_depth_metrics,
    valid_depth_mask,
)
from zed_da3_compare.ros_image_utils import (
    depth_msg_to_meters,
    depth_to_image_msg,
    depth_to_preview_msg,
    image_msg_to_rgb,
)


DEFAULT_DATA_DIR = Path("/home/usuario/depth_anything_ws/src/zed_da3_compare/data")
DEFAULT_RESULTS_DIR = Path("/home/usuario/depth_anything_ws/src/zed_da3_compare/results")
DEFAULT_BAG_CACHE_DIR = DEFAULT_RESULTS_DIR / "_bag_cache"


@dataclass
class BagRecord:
    stamp_ns: int
    bag_time_ns: int
    msg: Any


@dataclass
class SyncedTriple:
    index: int
    left: BagRecord
    right: BagRecord
    zed_depth: BagRecord

    @property
    def left_right_delta_ms(self) -> float:
        return abs(self.left.stamp_ns - self.right.stamp_ns) / 1e6

    @property
    def left_depth_delta_ms(self) -> float:
        return abs(self.left.stamp_ns - self.zed_depth.stamp_ns) / 1e6


class BagExporter:
    def __init__(self, bag_dir: Path, storage_id: str = "sqlite3") -> None:
        from rosbag2_py import ConverterOptions, SequentialWriter, StorageOptions, TopicMetadata

        from rclpy.serialization import serialize_message

        self._serialize_message = serialize_message
        self._writer = SequentialWriter()
        bag_dir.mkdir(parents=True, exist_ok=True)
        self._writer.open(
            StorageOptions(uri=str(bag_dir), storage_id=storage_id),
            ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
        )
        self._TopicMetadata = TopicMetadata
        self._created_topics: set[str] = set()

    def create_topic(self, name: str, type_name: str) -> None:
        if name in self._created_topics:
            return
        self._writer.create_topic(
            self._TopicMetadata(name=name, type=type_name, serialization_format="cdr", offered_qos_profiles="")
        )
        self._created_topics.add(name)

    def write(self, topic: str, msg: Any, timestamp_ns: int) -> None:
        serialized = self._serialize_message(msg)
        self._writer.write(topic, serialized, timestamp_ns)


def main() -> None:
    args = parse_args()
    os.environ.setdefault("DA3_LOG_LEVEL", args.da3_log_level)

    run_dir = prepare_run_dir(args)
    metadata: dict[str, Any] = {
        "args": vars(args),
        "bag_path": str(resolve_bag_path(args.bag)),
        "run_dir": str(run_dir),
    }
    print(f"[offline] results: {run_dir}")

    model = load_da3_model(args.model_dir, args.device)
    exporter = prepare_exporter(args, run_dir)
    if exporter is not None:
        metadata["export_bag_dir"] = str(exporter["bag_dir"])
        metadata["export_bag_storage_id"] = exporter["storage_id"]
        bag_exporter = exporter["writer"]
    else:
        bag_exporter = None
    rows: list[dict[str, float | int | str]] = []
    synced_seen = 0

    for triple in sync_bag_triples(args):
        if synced_seen % max(1, args.stride) != 0:
            synced_seen += 1
            continue
        out_idx = len(rows)
        row = evaluate_triple(args, run_dir, model, triple, out_idx, bag_exporter=bag_exporter)
        rows.append(row)
        synced_seen += 1
        print(
            "[offline] "
            f"evaluated={len(rows)} synced_seen={synced_seen} "
            f"stamp={row['stamp_ns']} "
            f"mono_scaled_abs_rel={row['mono_scaled_abs_rel']:.4f} "
            f"multicam_scaled_abs_rel={row['multicam_scaled_abs_rel']:.4f} "
            f"pair_corr={row['pair_corr']:.4f}"
        )
        if args.max_frames > 0 and len(rows) >= args.max_frames:
            break

    if not rows:
        raise RuntimeError(
            "No frames evaluated. Try increasing --sync-tolerance-ms or inspect topic names."
        )

    write_csv(run_dir / "metrics_per_frame.csv", rows)
    summary = build_summary(rows)
    metadata["synced_triples_seen"] = synced_seen
    metadata["evaluated_frames"] = len(rows)
    write_json(run_dir / "summary.json", summary)
    write_json(run_dir / "metadata.json", metadata)
    write_plots(run_dir, rows)
    print(f"[offline] wrote {len(rows)} evaluated frames")
    print(f"[offline] summary: {run_dir / 'summary.json'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", required=True, help="Bag path or name inside zed_da3_compare/data")
    parser.add_argument("--output-root", default=str(DEFAULT_RESULTS_DIR))
    parser.add_argument("--run-name", default="", help="Output folder name. Defaults to bag name + settings.")
    parser.add_argument("--model-dir", default=os.environ.get("DA3_MODEL_DIR", ""))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--da3-log-level", default="WARN", choices=["ERROR", "WARN", "INFO", "DEBUG"])
    parser.add_argument("--left-topic", default="/zed/zed_node/left/color/rect/image")
    parser.add_argument("--right-topic", default="/zed/zed_node/right/color/rect/image")
    parser.add_argument("--zed-depth-topic", default="/zed/zed_node/depth/depth_registered")
    parser.add_argument("--sync-tolerance-ms", type=float, default=40.0)
    parser.add_argument(
        "--zed-depth-time-offset-ms",
        type=float,
        default=0.0,
        help="Offset added to ZED depth header timestamps before synchronization.",
    )
    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument("--process-res-method", default="upper_bound_resize")
    parser.add_argument("--ref-view-strategy", default="saddle_balanced")
    parser.add_argument("--min-depth-m", type=float, default=0.2)
    parser.add_argument("--max-depth-m", type=float, default=20.0)
    parser.add_argument("--max-frames", type=int, default=0, help="0 means all synchronized frames")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--no-save-arrays", dest="save_arrays", action="store_false")
    parser.set_defaults(save_arrays=True)
    parser.add_argument("--save-visual-every", type=int, default=10, help="0 disables PNG visual dumps")
    parser.add_argument("--export-bag", action="store_true", help="Write a replayable rosbag with source inputs and DA3 outputs.")
    parser.add_argument("--export-bag-name", default="replay_bag", help="Subdirectory name inside the run folder for the exported bag.")
    parser.add_argument("--export-bag-storage-id", default="sqlite3", help="rosbag2 storage id for the exported bag.")
    parser.add_argument("--storage-id", default="", help="Override rosbag2 storage id, e.g. sqlite3 or mcap")
    return parser.parse_args()


def prepare_run_dir(args: argparse.Namespace) -> Path:
    bag_path = resolve_bag_path(args.bag)
    output_root = Path(args.output_root)
    run_name = args.run_name
    if not run_name:
        run_name = (
            f"{bag_path.name}_res{args.process_res}_tol{int(args.sync_tolerance_ms)}ms"
        )
    run_dir = output_root / run_name
    for subdir in [
        run_dir,
        run_dir / "arrays" / "zed",
        run_dir / "arrays" / "mono",
        run_dir / "arrays" / "multicam",
        run_dir / "visuals",
        run_dir / "plots",
    ]:
        subdir.mkdir(parents=True, exist_ok=True)
    return run_dir


def prepare_exporter(args: argparse.Namespace, run_dir: Path) -> dict[str, Any] | None:
    if not args.export_bag:
        return None
    bag_dir = next_available_bag_dir(run_dir / args.export_bag_name)
    writer = BagExporter(bag_dir, storage_id=args.export_bag_storage_id)
    create_export_topics(writer)
    print(f"[offline] exporting replay bag to: {bag_dir}")
    return {"bag_dir": bag_dir, "storage_id": args.export_bag_storage_id, "writer": writer}


def next_available_bag_dir(base_dir: Path) -> Path:
    if not base_dir.exists():
        return base_dir
    suffix = 1
    while True:
        candidate = base_dir.with_name(f"{base_dir.name}_{suffix:02d}")
        if not candidate.exists():
            return candidate
        suffix += 1


def create_export_topics(writer: BagExporter) -> None:
    writer.create_topic("/zed_da3_eval/left/color/rect/image", "sensor_msgs/msg/Image")
    writer.create_topic("/zed_da3_eval/right/color/rect/image", "sensor_msgs/msg/Image")
    writer.create_topic("/zed_da3_eval/depth/depth_registered", "sensor_msgs/msg/Image")
    writer.create_topic("/da3_compare/mono/depth/image", "sensor_msgs/msg/Image")
    writer.create_topic("/da3_compare/mono/preview", "sensor_msgs/msg/Image")
    writer.create_topic("/da3_compare/multicam/depth/image", "sensor_msgs/msg/Image")
    writer.create_topic("/da3_compare/multicam/preview", "sensor_msgs/msg/Image")


def resolve_bag_path(bag_arg: str) -> Path:
    bag_path = Path(bag_arg).expanduser()
    if bag_path.exists():
        return bag_path
    candidate = DEFAULT_DATA_DIR / bag_arg
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Bag not found: {bag_arg} or {candidate}")


def load_da3_model(model_dir: str, device: str):
    if not model_dir:
        raise ValueError("--model-dir is required or DA3_MODEL_DIR must be set")
    from depth_anything_3.api import DepthAnything3

    model_path = Path(model_dir).expanduser()
    if not model_path.is_dir():
        raise ValueError(f"Model directory does not exist: {model_path}")
    model = DepthAnything3.from_pretrained(str(model_path)).to(device).eval()
    print(f"[offline] loaded DA3 model: {model_path}")
    return model


def sync_bag_triples(args: argparse.Namespace):
    from rclpy.serialization import deserialize_message
    from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
    from rosidl_runtime_py.utilities import get_message

    bag_path = prepare_readable_bag_path(resolve_bag_path(args.bag))
    storage_id = args.storage_id or infer_storage_id(bag_path)
    reader = SequentialReader()
    reader.open(
        StorageOptions(uri=str(bag_path), storage_id=storage_id),
        ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )

    topic_types = {item.name: item.type for item in reader.get_all_topics_and_types()}
    required = [args.left_topic, args.right_topic, args.zed_depth_topic]
    missing = [topic for topic in required if topic not in topic_types]
    if missing:
        raise RuntimeError(f"Missing topics in bag: {missing}. Available: {sorted(topic_types)}")
    msg_types = {topic: get_message(topic_types[topic]) for topic in required}

    buffers: dict[str, list[BagRecord]] = {topic: [] for topic in required}
    tolerance_ns = int(args.sync_tolerance_ms * 1e6)
    depth_offset_ns = int(args.zed_depth_time_offset_ms * 1e6)
    matched = 0
    counts = {topic: 0 for topic in required}

    while reader.has_next():
        topic, serialized, bag_time_ns = reader.read_next()
        if topic not in msg_types:
            continue
        msg = deserialize_message(serialized, msg_types[topic])
        stamp_ns = message_stamp_ns(msg, bag_time_ns)
        if topic == args.zed_depth_topic:
            stamp_ns += depth_offset_ns
        buffers[topic].append(BagRecord(stamp_ns=stamp_ns, bag_time_ns=bag_time_ns, msg=msg))
        counts[topic] += 1

        while True:
            triple = pop_next_match(buffers, args.left_topic, args.right_topic, args.zed_depth_topic, tolerance_ns)
            if triple is None:
                break
            yield SyncedTriple(index=matched, left=triple[0], right=triple[1], zed_depth=triple[2])
            matched += 1

        prune_old_records(buffers, current_stamp_ns=stamp_ns, keep_window_ns=max(tolerance_ns * 10, int(2e9)))

    print(f"[offline] bag counts: {counts}")
    print(f"[offline] synchronized triples: {matched}")


def message_stamp_ns(msg, fallback_ns: int) -> int:
    header = getattr(msg, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return int(fallback_ns)
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def pop_next_match(
    buffers: dict[str, list[BagRecord]],
    left_topic: str,
    right_topic: str,
    depth_topic: str,
    tolerance_ns: int,
) -> tuple[BagRecord, BagRecord, BagRecord] | None:
    left_buffer = buffers[left_topic]
    if not left_buffer or not buffers[right_topic] or not buffers[depth_topic]:
        return None

    left = left_buffer[0]
    right_idx, right = nearest_record(buffers[right_topic], left.stamp_ns)
    depth_idx, depth = nearest_record(buffers[depth_topic], left.stamp_ns)
    if right is None or depth is None:
        return None
    if abs(right.stamp_ns - left.stamp_ns) > tolerance_ns:
        return None
    if abs(depth.stamp_ns - left.stamp_ns) > tolerance_ns:
        return None

    left_buffer.pop(0)
    buffers[right_topic].pop(right_idx)
    buffers[depth_topic].pop(depth_idx)
    return left, right, depth


def nearest_record(records: list[BagRecord], stamp_ns: int) -> tuple[int, BagRecord | None]:
    if not records:
        return -1, None
    deltas = [abs(record.stamp_ns - stamp_ns) for record in records]
    idx = int(np.argmin(deltas))
    return idx, records[idx]


def prune_old_records(
    buffers: dict[str, list[BagRecord]],
    *,
    current_stamp_ns: int,
    keep_window_ns: int,
) -> None:
    min_stamp = current_stamp_ns - keep_window_ns
    for topic, records in buffers.items():
        first_keep = 0
        while first_keep < len(records) and records[first_keep].stamp_ns < min_stamp:
            first_keep += 1
        if first_keep:
            del records[:first_keep]


def infer_storage_id(bag_path: Path) -> str:
    metadata = bag_path / "metadata.yaml"
    if not metadata.exists():
        return "sqlite3"
    text = metadata.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"storage_identifier:\s*([^\s]+)", text)
    if match:
        return match.group(1).strip()
    if list(bag_path.glob("*.mcap")):
        return "mcap"
    return "sqlite3"


def prepare_readable_bag_path(bag_path: Path) -> Path:
    metadata = read_bag_metadata_text(bag_path)
    compression_format = read_metadata_value(metadata, "compression_format")
    compression_mode = read_metadata_value(metadata, "compression_mode")
    if not compression_format:
        return bag_path
    if compression_format != "zstd" or compression_mode != "FILE":
        raise RuntimeError(
            "Unsupported compressed bag. Expected compression_format=zstd and "
            f"compression_mode=FILE, got format={compression_format!r} mode={compression_mode!r}"
        )

    readable_path = DEFAULT_BAG_CACHE_DIR / f"{bag_path.name}_decompressed"
    source_files = sorted(bag_path.glob("*.zstd"))
    if not source_files:
        raise RuntimeError(f"Compressed bag metadata found but no .zstd files exist in {bag_path}")

    readable_path.mkdir(parents=True, exist_ok=True)
    for source_file in source_files:
        output_file = readable_path / source_file.name.removesuffix(".zstd")
        if not output_file.exists() or output_file.stat().st_size == 0:
            decompress_zstd_file(source_file, output_file)

    rewritten_metadata = rewrite_file_compressed_metadata(metadata)
    (readable_path / "metadata.yaml").write_text(rewritten_metadata, encoding="utf-8")
    print(f"[offline] decompressed bag cache: {readable_path}")
    return readable_path


def read_bag_metadata_text(bag_path: Path) -> str:
    metadata_path = bag_path / "metadata.yaml"
    if not metadata_path.exists():
        return ""
    return metadata_path.read_text(encoding="utf-8", errors="replace")


def read_metadata_value(metadata: str, key: str) -> str:
    match = re.search(rf"^\s*{re.escape(key)}:\s*([^\s]+)\s*$", metadata, flags=re.MULTILINE)
    if not match:
        return ""
    value = match.group(1).strip()
    if value in {"''", '""'}:
        return ""
    return value


def decompress_zstd_file(source_file: Path, output_file: Path) -> None:
    import zstandard

    with source_file.open("rb") as src, output_file.open("wb") as dst:
        reader = zstandard.ZstdDecompressor().stream_reader(src)
        shutil.copyfileobj(reader, dst)


def rewrite_file_compressed_metadata(metadata: str) -> str:
    text = re.sub(r"^(\s*)compression_format:\s*[^\s]+\s*$", r"\1compression_format: ''", metadata, flags=re.MULTILINE)
    text = re.sub(r"^(\s*)compression_mode:\s*[^\s]+\s*$", r"\1compression_mode: ''", text, flags=re.MULTILINE)
    text = text.replace(".db3.zstd", ".db3")
    text = text.replace(".mcap.zstd", ".mcap")
    return text


def evaluate_triple(
    args: argparse.Namespace,
    run_dir: Path,
    model,
    triple: SyncedTriple,
    out_idx: int,
    *,
    bag_exporter: BagExporter | None = None,
) -> dict[str, Any]:
    left_rgb = image_msg_to_rgb(triple.left.msg)
    right_rgb = image_msg_to_rgb(triple.right.msg)
    zed_depth = depth_msg_to_meters(triple.zed_depth.msg)

    with torch.inference_mode():
        mono_prediction = model.inference(
            image=[left_rgb],
            process_res=args.process_res,
            process_res_method=args.process_res_method,
            export_dir=None,
        )
        multicam_prediction = model.inference(
            image=[left_rgb, right_rgb],
            process_res=args.process_res,
            process_res_method=args.process_res_method,
            ref_view_strategy=args.ref_view_strategy,
            export_dir=None,
        )

    mono_depth = resize_depth_to_shape(np.asarray(mono_prediction.depth[0], dtype=np.float32), zed_depth.shape[:2])
    multicam_depth = resize_depth_to_shape(np.asarray(multicam_prediction.depth[0], dtype=np.float32), zed_depth.shape[:2])
    zed_depth = np.asarray(zed_depth, dtype=np.float32)
    mono_depth_msg = depth_to_image_msg(mono_depth, triple.left.msg)
    multicam_depth_msg = depth_to_image_msg(multicam_depth, triple.left.msg)
    mono_preview_msg = depth_to_preview_msg(mono_depth, triple.left.msg)
    multicam_preview_msg = depth_to_preview_msg(multicam_depth, triple.left.msg)

    row = compute_frame_row(args, triple, out_idx, mono_prediction, multicam_prediction, mono_depth, multicam_depth, zed_depth)
    stem = f"{out_idx:06d}"
    if args.save_arrays:
        np.save(run_dir / "arrays" / "zed" / f"{stem}.npy", zed_depth)
        np.save(run_dir / "arrays" / "mono" / f"{stem}.npy", mono_depth)
        np.save(run_dir / "arrays" / "multicam" / f"{stem}.npy", multicam_depth)
    if args.save_visual_every > 0 and out_idx % args.save_visual_every == 0:
        write_visuals(run_dir / "visuals", stem, left_rgb, right_rgb, zed_depth, mono_depth, multicam_depth, args)
    if bag_exporter is not None:
        export_frame_bag(
            bag_exporter,
            triple,
            mono_depth_msg,
            mono_preview_msg,
            multicam_depth_msg,
            multicam_preview_msg,
        )
    return row


def export_frame_bag(
    bag_exporter: BagExporter,
    triple: SyncedTriple,
    mono_depth_msg,
    mono_preview_msg,
    multicam_depth_msg,
    multicam_preview_msg,
) -> None:
    bag_exporter.write("/zed_da3_eval/left/color/rect/image", triple.left.msg, triple.left.bag_time_ns)
    bag_exporter.write("/zed_da3_eval/right/color/rect/image", triple.right.msg, triple.right.bag_time_ns)
    bag_exporter.write("/zed_da3_eval/depth/depth_registered", triple.zed_depth.msg, triple.zed_depth.bag_time_ns)

    bag_exporter.write("/da3_compare/mono/depth/image", mono_depth_msg, triple.left.bag_time_ns)
    bag_exporter.write("/da3_compare/mono/preview", mono_preview_msg, triple.left.bag_time_ns)
    bag_exporter.write("/da3_compare/multicam/depth/image", multicam_depth_msg, triple.left.bag_time_ns)
    bag_exporter.write("/da3_compare/multicam/preview", multicam_preview_msg, triple.left.bag_time_ns)


def compute_frame_row(
    args: argparse.Namespace,
    triple: SyncedTriple,
    out_idx: int,
    mono_prediction,
    multicam_prediction,
    mono_depth: np.ndarray,
    multicam_depth: np.ndarray,
    zed_depth: np.ndarray,
) -> dict[str, Any]:
    ref_mask = valid_depth_mask(zed_depth, min_depth_m=args.min_depth_m, max_depth_m=args.max_depth_m)
    mono_mask = ref_mask & np.isfinite(mono_depth) & (mono_depth > 0.0)
    multicam_mask = ref_mask & np.isfinite(multicam_depth) & (multicam_depth > 0.0)
    pair_mask = joint_valid_mask(
        mono_depth,
        multicam_depth,
        min_depth_m=args.min_depth_m,
        max_depth_m=args.max_depth_m,
    )

    mono_raw = supervised_depth_metrics(mono_depth, zed_depth, mono_mask)
    mono_scale = median_scale_to_reference(mono_depth, zed_depth, mono_mask)
    mono_scaled = supervised_depth_metrics(mono_depth * mono_scale, zed_depth, mono_mask)

    multicam_raw = supervised_depth_metrics(multicam_depth, zed_depth, multicam_mask)
    multicam_scale = median_scale_to_reference(multicam_depth, zed_depth, multicam_mask)
    multicam_scaled = supervised_depth_metrics(multicam_depth * multicam_scale, zed_depth, multicam_mask)

    pair = pairwise_depth_metrics(mono_depth, multicam_depth, pair_mask)

    row: dict[str, Any] = {
        "frame_idx": out_idx,
        "source_idx": triple.index,
        "stamp_ns": triple.left.stamp_ns,
        "left_right_delta_ms": triple.left_right_delta_ms,
        "left_depth_delta_ms": triple.left_depth_delta_ms,
        "valid_zed_ratio": float(np.mean(ref_mask)),
        "valid_mono_ratio": float(np.mean(mono_mask)),
        "valid_multicam_ratio": float(np.mean(multicam_mask)),
        "valid_pair_ratio": float(np.mean(pair_mask)),
        "mono_is_metric": int(getattr(mono_prediction, "is_metric", 0)),
        "multicam_is_metric": int(getattr(multicam_prediction, "is_metric", 0)),
        "mono_model_scale_factor": float_or_nan(getattr(mono_prediction, "scale_factor", float("nan"))),
        "multicam_model_scale_factor": float_or_nan(getattr(multicam_prediction, "scale_factor", float("nan"))),
        "mono_median_scale_to_zed": mono_scale,
        "multicam_median_scale_to_zed": multicam_scale,
    }
    row.update(prefix_dict("mono_raw", mono_raw))
    row.update(prefix_dict("mono_scaled", mono_scaled))
    row.update(prefix_dict("multicam_raw", multicam_raw))
    row.update(prefix_dict("multicam_scaled", multicam_scaled))
    row.update(prefix_dict("pair", pair))
    row["winner_raw_abs_rel"] = winner(row, "mono_raw_abs_rel", "multicam_raw_abs_rel")
    row["winner_scaled_abs_rel"] = winner(row, "mono_scaled_abs_rel", "multicam_scaled_abs_rel")
    row["winner_raw_rmse"] = winner(row, "mono_raw_rmse", "multicam_raw_rmse")
    row["winner_scaled_rmse"] = winner(row, "mono_scaled_rmse", "multicam_scaled_rmse")
    return row


def prefix_dict(prefix: str, values: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def winner(row: dict[str, Any], mono_key: str, multicam_key: str) -> str:
    mono = float(row.get(mono_key, float("nan")))
    multicam = float(row.get(multicam_key, float("nan")))
    if not np.isfinite(mono) and not np.isfinite(multicam):
        return "none"
    if not np.isfinite(mono):
        return "multicam"
    if not np.isfinite(multicam):
        return "mono"
    return "mono" if mono <= multicam else "multicam"


def float_or_nan(value) -> float:
    try:
        if value is None:
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def write_visuals(
    visual_dir: Path,
    stem: str,
    left_rgb: np.ndarray,
    right_rgb: np.ndarray,
    zed_depth: np.ndarray,
    mono_depth: np.ndarray,
    multicam_depth: np.ndarray,
    args: argparse.Namespace,
) -> None:
    cv2.imwrite(str(visual_dir / f"{stem}_left.png"), cv2.cvtColor(left_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(visual_dir / f"{stem}_right.png"), cv2.cvtColor(right_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(visual_dir / f"{stem}_zed_depth.png"), colorize_depth(zed_depth, args.min_depth_m, args.max_depth_m))
    cv2.imwrite(str(visual_dir / f"{stem}_mono_depth.png"), colorize_depth(mono_depth, args.min_depth_m, args.max_depth_m))
    cv2.imwrite(str(visual_dir / f"{stem}_multicam_depth.png"), colorize_depth(multicam_depth, args.min_depth_m, args.max_depth_m))
    cv2.imwrite(str(visual_dir / f"{stem}_mono_abs_error.png"), colorize_error(np.abs(mono_depth - zed_depth), max_error=2.0))
    cv2.imwrite(str(visual_dir / f"{stem}_multicam_abs_error.png"), colorize_error(np.abs(multicam_depth - zed_depth), max_error=2.0))
    cv2.imwrite(str(visual_dir / f"{stem}_mono_multicam_abs_diff.png"), colorize_error(np.abs(mono_depth - multicam_depth), max_error=2.0))


def colorize_depth(depth_m: np.ndarray, min_depth: float, max_depth: float) -> np.ndarray:
    valid = np.isfinite(depth_m) & (depth_m > 0.0)
    norm = np.zeros(depth_m.shape, dtype=np.uint8)
    if valid.any():
        clipped = np.clip(depth_m, min_depth, max_depth)
        norm = ((clipped - min_depth) / max(max_depth - min_depth, 1e-6) * 255.0).astype(np.uint8)
        norm[~valid] = 0
    return cv2.applyColorMap(255 - norm, cv2.COLORMAP_TURBO)


def colorize_error(error_m: np.ndarray, max_error: float) -> np.ndarray:
    valid = np.isfinite(error_m)
    norm = np.zeros(error_m.shape, dtype=np.uint8)
    if valid.any():
        norm = (np.clip(error_m, 0.0, max_error) / max(max_error, 1e-6) * 255.0).astype(np.uint8)
        norm[~valid] = 0
    return cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def build_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    numeric_keys = [
        key
        for key, value in rows[0].items()
        if isinstance(value, (int, float, np.integer, np.floating))
    ]
    summary = {
        "num_frames": len(rows),
        "metrics": summarize_metric_rows(rows, numeric_keys),
        "winner_counts": {},
    }
    for key in ["winner_raw_abs_rel", "winner_scaled_abs_rel", "winner_raw_rmse", "winner_scaled_rmse"]:
        counts: dict[str, int] = {}
        for row in rows:
            value = str(row.get(key, "none"))
            counts[value] = counts.get(value, 0) + 1
        summary["winner_counts"][key] = counts
    return summary


def write_json(path: Path, data: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def write_plots(run_dir: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = run_dir / "plots"
    x = np.asarray([row["frame_idx"] for row in rows], dtype=np.float64)

    plot_lines(
        plots_dir / "abs_rel_over_frames.png",
        x,
        rows,
        [
            ("mono_raw_abs_rel", "mono raw"),
            ("multicam_raw_abs_rel", "multicam raw"),
            ("mono_scaled_abs_rel", "mono scaled"),
            ("multicam_scaled_abs_rel", "multicam scaled"),
        ],
        ylabel="abs_rel",
        title="Absolute relative error vs ZED",
    )
    plot_lines(
        plots_dir / "rmse_over_frames.png",
        x,
        rows,
        [
            ("mono_raw_rmse", "mono raw"),
            ("multicam_raw_rmse", "multicam raw"),
            ("mono_scaled_rmse", "mono scaled"),
            ("multicam_scaled_rmse", "multicam scaled"),
        ],
        ylabel="RMSE [m]",
        title="RMSE vs ZED",
    )
    plot_lines(
        plots_dir / "pairwise_over_frames.png",
        x,
        rows,
        [
            ("pair_mae", "mono/multicam MAE"),
            ("pair_rmse", "mono/multicam RMSE"),
            ("pair_grad_mae", "gradient MAE"),
        ],
        ylabel="difference",
        title="DA3 mono vs multiview difference",
    )
    plot_lines(
        plots_dir / "scale_over_frames.png",
        x,
        rows,
        [
            ("mono_median_scale_to_zed", "mono scale to ZED"),
            ("multicam_median_scale_to_zed", "multicam scale to ZED"),
            ("pair_median_scale_b_over_a", "multicam/mono scale"),
        ],
        ylabel="scale",
        title="Median scale factors",
    )
    plot_hist(
        plots_dir / "scaled_abs_rel_hist.png",
        rows,
        ["mono_scaled_abs_rel", "multicam_scaled_abs_rel"],
        ["mono scaled", "multicam scaled"],
        xlabel="scaled abs_rel",
        title="Scaled abs_rel distribution",
    )
    plt.close("all")


def plot_lines(path: Path, x: np.ndarray, rows: list[dict[str, Any]], series, ylabel: str, title: str) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 5))
    for key, label in series:
        y = np.asarray([float(row.get(key, float("nan"))) for row in rows], dtype=np.float64)
        plt.plot(x, y, label=label, linewidth=1.4)
    plt.xlabel("frame")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def plot_hist(path: Path, rows: list[dict[str, Any]], keys, labels, xlabel: str, title: str) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(9, 5))
    for key, label in zip(keys, labels):
        values = np.asarray([float(row.get(key, float("nan"))) for row in rows], dtype=np.float64)
        values = values[np.isfinite(values)]
        plt.hist(values, bins=40, alpha=0.55, label=label)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


if __name__ == "__main__":
    main()
