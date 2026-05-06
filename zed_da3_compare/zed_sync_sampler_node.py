#!/usr/bin/env python3
"""Publish low-rate synchronized ZED RGB/depth triples for offline evaluation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image

from zed_da3_compare.da3_common import stamp_to_seconds


@dataclass(frozen=True)
class Match:
    left: Image
    right: Image
    depth: Image
    left_right_delta_ms: float
    left_depth_delta_ms: float


class ZedSyncSamplerNode(Node):
    def __init__(self) -> None:
        super().__init__("zed_sync_sampler_node")

        self.declare_parameter("left_image_topic", "/zed/zed_node/left/color/rect/image")
        self.declare_parameter("right_image_topic", "/zed/zed_node/right/color/rect/image")
        self.declare_parameter("depth_topic", "/zed/zed_node/depth/depth_registered")
        self.declare_parameter("left_camera_info_topic", "/zed/zed_node/left/color/rect/camera_info")
        self.declare_parameter("right_camera_info_topic", "/zed/zed_node/right/color/rect/camera_info")
        self.declare_parameter("output_left_image_topic", "/zed_da3_eval/left/color/rect/image")
        self.declare_parameter("output_right_image_topic", "/zed_da3_eval/right/color/rect/image")
        self.declare_parameter("output_depth_topic", "/zed_da3_eval/depth/depth_registered")
        self.declare_parameter("output_left_camera_info_topic", "/zed_da3_eval/left/color/rect/camera_info")
        self.declare_parameter("output_right_camera_info_topic", "/zed_da3_eval/right/color/rect/camera_info")
        self.declare_parameter("sample_rate_hz", 2.0)
        self.declare_parameter("sync_tolerance_ms", 20.0)
        self.declare_parameter("queue_size", 120)
        self.declare_parameter("log_every_n", 10)

        queue_size = max(3, int(self.get_parameter("queue_size").value))
        self.left_queue: Deque[Image] = deque(maxlen=queue_size)
        self.right_queue: Deque[Image] = deque(maxlen=queue_size)
        self.depth_queue: Deque[Image] = deque(maxlen=queue_size)
        self.left_info: CameraInfo | None = None
        self.right_info: CameraInfo | None = None
        self.last_published_left_key: tuple[int, int] | None = None
        self.published_count = 0
        self.missed_count = 0

        left_topic = str(self.get_parameter("left_image_topic").value)
        right_topic = str(self.get_parameter("right_image_topic").value)
        depth_topic = str(self.get_parameter("depth_topic").value)
        left_info_topic = str(self.get_parameter("left_camera_info_topic").value)
        right_info_topic = str(self.get_parameter("right_camera_info_topic").value)
        out_left_topic = str(self.get_parameter("output_left_image_topic").value)
        out_right_topic = str(self.get_parameter("output_right_image_topic").value)
        out_depth_topic = str(self.get_parameter("output_depth_topic").value)
        out_left_info_topic = str(self.get_parameter("output_left_camera_info_topic").value)
        out_right_info_topic = str(self.get_parameter("output_right_camera_info_topic").value)

        self.left_pub = self.create_publisher(Image, out_left_topic, 1)
        self.right_pub = self.create_publisher(Image, out_right_topic, 1)
        self.depth_pub = self.create_publisher(Image, out_depth_topic, 1)
        self.left_info_pub = self.create_publisher(CameraInfo, out_left_info_topic, 1)
        self.right_info_pub = self.create_publisher(CameraInfo, out_right_info_topic, 1)

        self.create_subscription(Image, left_topic, self.on_left_image, 10)
        self.create_subscription(Image, right_topic, self.on_right_image, 10)
        self.create_subscription(Image, depth_topic, self.on_depth, 10)
        self.create_subscription(CameraInfo, left_info_topic, self.on_left_info, 10)
        self.create_subscription(CameraInfo, right_info_topic, self.on_right_info, 10)

        sample_rate_hz = max(0.1, float(self.get_parameter("sample_rate_hz").value))
        self.create_timer(1.0 / sample_rate_hz, self.on_timer)

        self.get_logger().info(f"Sub left:  {left_topic}")
        self.get_logger().info(f"Sub right: {right_topic}")
        self.get_logger().info(f"Sub depth: {depth_topic}")
        self.get_logger().info(f"Pub left:  {out_left_topic}")
        self.get_logger().info(f"Pub right: {out_right_topic}")
        self.get_logger().info(f"Pub depth: {out_depth_topic}")
        self.get_logger().info(
            f"Sampling synchronized triples at {sample_rate_hz:.2f} Hz "
            f"with tolerance {float(self.get_parameter('sync_tolerance_ms').value):.1f} ms"
        )

    def on_left_image(self, msg: Image) -> None:
        self.left_queue.append(msg)

    def on_right_image(self, msg: Image) -> None:
        self.right_queue.append(msg)

    def on_depth(self, msg: Image) -> None:
        self.depth_queue.append(msg)

    def on_left_info(self, msg: CameraInfo) -> None:
        self.left_info = msg

    def on_right_info(self, msg: CameraInfo) -> None:
        self.right_info = msg

    def on_timer(self) -> None:
        match = self.find_latest_match()
        if match is None:
            self.missed_count += 1
            if self.missed_count % max(1, int(self.get_parameter("log_every_n").value)) == 0:
                self.get_logger().warn(
                    "No synchronized triple available "
                    f"(left={len(self.left_queue)} right={len(self.right_queue)} depth={len(self.depth_queue)})"
                )
            return

        left_key = (match.left.header.stamp.sec, match.left.header.stamp.nanosec)
        if left_key == self.last_published_left_key:
            return
        self.last_published_left_key = left_key

        self.left_pub.publish(match.left)
        self.right_pub.publish(match.right)
        self.depth_pub.publish(match.depth)
        if self.left_info is not None:
            self.left_info_pub.publish(self.left_info)
        if self.right_info is not None:
            self.right_info_pub.publish(self.right_info)

        self.published_count += 1
        if self.published_count % max(1, int(self.get_parameter("log_every_n").value)) == 0:
            self.get_logger().info(
                f"Published {self.published_count} synced triples "
                f"left/right={match.left_right_delta_ms:.2f} ms "
                f"left/depth={match.left_depth_delta_ms:.2f} ms"
            )

    def find_latest_match(self) -> Match | None:
        if not self.left_queue or not self.right_queue or not self.depth_queue:
            return None

        tolerance_ms = float(self.get_parameter("sync_tolerance_ms").value)
        for left in reversed(self.left_queue):
            right, right_delta_ms = nearest_by_stamp(left, self.right_queue)
            depth, depth_delta_ms = nearest_by_stamp(left, self.depth_queue)
            if right is None or depth is None:
                continue
            if right_delta_ms <= tolerance_ms and depth_delta_ms <= tolerance_ms:
                return Match(
                    left=left,
                    right=right,
                    depth=depth,
                    left_right_delta_ms=right_delta_ms,
                    left_depth_delta_ms=depth_delta_ms,
                )
        return None


def nearest_by_stamp(reference: Image, candidates: Deque[Image]) -> tuple[Image | None, float]:
    if not candidates:
        return None, float("inf")
    ref_t = stamp_to_seconds(reference)
    best = min(candidates, key=lambda msg: abs(stamp_to_seconds(msg) - ref_t))
    delta_ms = abs(stamp_to_seconds(best) - ref_t) * 1000.0
    return best, delta_ms


def main() -> None:
    rclpy.init()
    node = ZedSyncSamplerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
