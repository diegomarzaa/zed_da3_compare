#!/usr/bin/env python3
"""Compare DA3 mono and multiview depth from the same synchronized color images."""

from __future__ import annotations

import cv2
import numpy as np
import rclpy
import torch
from rclpy.node import Node
from sensor_msgs.msg import Image

from zed_da3_compare.da3_common import load_da3_model, log_depth_stats, stamp_to_seconds
from zed_da3_compare.depth_metrics import (
    median_scale_to_reference,
    pairwise_depth_metrics,
    resize_depth_to_shape,
    supervised_depth_metrics,
)
from zed_da3_compare.ros_image_utils import (
    depth_msg_to_meters,
    depth_to_image_msg,
    depth_to_preview_msg,
    image_msg_to_rgb,
)


class Da3DepthMulticamNode(Node):
    def __init__(self) -> None:
        super().__init__("da3_depth_multicam_node")

        self.declare_parameter("left_image_topic", "/zed/zed_node/left/color/rect/image")
        self.declare_parameter("right_image_topic", "/zed/zed_node/right/color/rect/image")
        self.declare_parameter("mono_output_depth_topic", "/da3_compare/mono/depth/image")
        self.declare_parameter("mono_output_preview_topic", "/da3_compare/mono/preview")
        self.declare_parameter("multicam_output_depth_topic", "/da3_compare/multicam/depth/image")
        self.declare_parameter("multicam_output_preview_topic", "/da3_compare/multicam/preview")
        self.declare_parameter("model_dir", "")
        self.declare_parameter("device", "cuda")
        self.declare_parameter("process_res", 504)
        self.declare_parameter("process_res_method", "upper_bound_resize")
        self.declare_parameter("process_every_n", 1)
        self.declare_parameter("match_input_size", True)
        self.declare_parameter("max_stamp_delta_ms", 40.0)
        self.declare_parameter("zed_depth_topic", "/zed/zed_node/depth/depth_registered")
        self.declare_parameter("enable_metrics", True)
        self.declare_parameter("metrics_every_n", 10)
        self.declare_parameter("max_depth_stamp_delta_ms", 80.0)
        self.declare_parameter("min_eval_depth_m", 0.2)
        self.declare_parameter("max_eval_depth_m", 20.0)
        self.declare_parameter("ref_view_strategy", "saddle_balanced")
        self.declare_parameter("preview_low_percentile", 1.0)
        self.declare_parameter("preview_high_percentile", 99.0)
        self.declare_parameter("preview_use_inverse_depth", True)

        self.device = str(self.get_parameter("device").value)
        self.model = load_da3_model(self)

        self.left_msg: Image | None = None
        self.right_msg: Image | None = None
        self.zed_depth_msg: Image | None = None
        self.frame_count = 0
        self.last_pair_key: tuple[int, int, int, int] | None = None

        left_image_topic = str(self.get_parameter("left_image_topic").value)
        right_image_topic = str(self.get_parameter("right_image_topic").value)
        mono_depth_topic = str(self.get_parameter("mono_output_depth_topic").value)
        mono_preview_topic = str(self.get_parameter("mono_output_preview_topic").value)
        multicam_depth_topic = str(self.get_parameter("multicam_output_depth_topic").value)
        multicam_preview_topic = str(self.get_parameter("multicam_output_preview_topic").value)
        zed_depth_topic = str(self.get_parameter("zed_depth_topic").value)

        self.mono_depth_pub = self.create_publisher(Image, mono_depth_topic, 1)
        self.mono_preview_pub = self.create_publisher(Image, mono_preview_topic, 1)
        self.multicam_depth_pub = self.create_publisher(Image, multicam_depth_topic, 1)
        self.multicam_preview_pub = self.create_publisher(Image, multicam_preview_topic, 1)
        self.create_subscription(Image, left_image_topic, self.on_left_image, 1)
        self.create_subscription(Image, right_image_topic, self.on_right_image, 1)
        self.create_subscription(Image, zed_depth_topic, self.on_zed_depth, 1)

        self.get_logger().info(f"Subscribed left image:  {left_image_topic}")
        self.get_logger().info(f"Subscribed right image: {right_image_topic}")
        self.get_logger().info(f"Subscribed ZED depth:  {zed_depth_topic}")
        self.get_logger().info(f"Publishing: {mono_depth_topic} [sensor_msgs/Image, 32FC1, mono left depth]")
        self.get_logger().info(f"Publishing: {mono_preview_topic} [sensor_msgs/Image, rgb8, mono preview]")
        self.get_logger().info(f"Publishing: {multicam_depth_topic} [sensor_msgs/Image, 32FC1, multiview left depth]")
        self.get_logger().info(f"Publishing: {multicam_preview_topic} [sensor_msgs/Image, rgb8, multiview preview]")
        self.get_logger().info("DA3 compare mode: one model, two sequential inference calls per synchronized pair")

    def on_left_image(self, msg: Image) -> None:
        self.left_msg = msg
        self.try_process_pair()

    def on_right_image(self, msg: Image) -> None:
        self.right_msg = msg
        self.try_process_pair()

    def on_zed_depth(self, msg: Image) -> None:
        self.zed_depth_msg = msg

    def try_process_pair(self) -> None:
        if self.left_msg is None or self.right_msg is None:
            return

        max_delta_s = float(self.get_parameter("max_stamp_delta_ms").value) / 1000.0
        left_t = stamp_to_seconds(self.left_msg)
        right_t = stamp_to_seconds(self.right_msg)
        if abs(left_t - right_t) > max_delta_s:
            return

        pair_key = (
            self.left_msg.header.stamp.sec,
            self.left_msg.header.stamp.nanosec,
            self.right_msg.header.stamp.sec,
            self.right_msg.header.stamp.nanosec,
        )
        if pair_key == self.last_pair_key:
            return
        self.last_pair_key = pair_key

        self.frame_count += 1
        process_every_n = int(self.get_parameter("process_every_n").value)
        if self.frame_count % max(1, process_every_n) != 0:
            return

        left_rgb = image_msg_to_rgb(self.left_msg)
        right_rgb = image_msg_to_rgb(self.right_msg)
        process_res = int(self.get_parameter("process_res").value)
        process_res_method = str(self.get_parameter("process_res_method").value)
        ref_view_strategy = str(self.get_parameter("ref_view_strategy").value)

        with torch.inference_mode():
            mono_prediction = self.model.inference(
                image=[left_rgb],
                process_res=process_res,
                process_res_method=process_res_method,
                export_dir=None,
            )
            multicam_prediction = self.model.inference(
                image=[left_rgb, right_rgb],
                process_res=process_res,
                process_res_method=process_res_method,
                ref_view_strategy=ref_view_strategy,
                export_dir=None,
            )

        mono_left_depth_m = np.asarray(mono_prediction.depth[0], dtype=np.float32)
        multicam_left_depth_m = np.asarray(multicam_prediction.depth[0], dtype=np.float32)
        if self.frame_count % 30 == 0:
            log_depth_stats(self.get_logger(), "DA3 mono left", mono_prediction, mono_left_depth_m)
            log_depth_stats(self.get_logger(), "DA3 multicam left", multicam_prediction, multicam_left_depth_m)

        if bool(self.get_parameter("match_input_size").value):
            mono_left_depth_m = cv2.resize(
                mono_left_depth_m,
                (self.left_msg.width, self.left_msg.height),
                interpolation=cv2.INTER_LINEAR,
            )
            multicam_left_depth_m = cv2.resize(
                multicam_left_depth_m,
                (self.left_msg.width, self.left_msg.height),
                interpolation=cv2.INTER_LINEAR,
            )

        self.mono_depth_pub.publish(depth_to_image_msg(mono_left_depth_m, self.left_msg))
        self.mono_preview_pub.publish(
            depth_to_preview_msg(
                mono_left_depth_m,
                self.left_msg,
                low_percentile=float(self.get_parameter("preview_low_percentile").value),
                high_percentile=float(self.get_parameter("preview_high_percentile").value),
                use_inverse_depth=bool(self.get_parameter("preview_use_inverse_depth").value),
            )
        )
        self.multicam_depth_pub.publish(depth_to_image_msg(multicam_left_depth_m, self.left_msg))
        self.multicam_preview_pub.publish(
            depth_to_preview_msg(
                multicam_left_depth_m,
                self.left_msg,
                low_percentile=float(self.get_parameter("preview_low_percentile").value),
                high_percentile=float(self.get_parameter("preview_high_percentile").value),
                use_inverse_depth=bool(self.get_parameter("preview_use_inverse_depth").value),
            )
        )

        self.maybe_log_metrics(mono_left_depth_m, multicam_left_depth_m)

    def maybe_log_metrics(self, mono_depth_m: np.ndarray, multicam_depth_m: np.ndarray) -> None:
        if not bool(self.get_parameter("enable_metrics").value):
            return

        metrics_every_n = max(1, int(self.get_parameter("metrics_every_n").value))
        if self.frame_count % metrics_every_n != 0:
            return

        valid_pair = (
            np.isfinite(mono_depth_m)
            & np.isfinite(multicam_depth_m)
            & (mono_depth_m > 0.0)
            & (multicam_depth_m > 0.0)
        )
        if valid_pair.any():
            pair = pairwise_depth_metrics(mono_depth_m, multicam_depth_m, valid_pair)
            self.get_logger().info(
                "DA3 mono vs multicam "
                f"valid={valid_pair.mean() * 100.0:.1f}% "
                f"mae={pair['mae']:.3f}m "
                f"mean_rel={pair['mean_rel_to_a']:.3f} "
                f"median_scale_multi_over_mono={pair['median_scale_b_over_a']:.3f} "
                f"corr={pair['corr']:.3f}"
            )

        if self.zed_depth_msg is None:
            self.get_logger().warn("Depth metrics against ZED skipped: no ZED depth message received yet")
            return

        max_delta_s = float(self.get_parameter("max_depth_stamp_delta_ms").value) / 1000.0
        left_t = stamp_to_seconds(self.left_msg)
        zed_t = stamp_to_seconds(self.zed_depth_msg)
        if abs(left_t - zed_t) > max_delta_s:
            self.get_logger().warn(
                "Depth metrics against ZED skipped: "
                f"left/ZED stamp delta is {abs(left_t - zed_t) * 1000.0:.1f} ms"
            )
            return

        zed_depth_m = depth_msg_to_meters(self.zed_depth_msg)
        mono_eval = resize_depth_to_shape(mono_depth_m, zed_depth_m.shape[:2])
        multicam_eval = resize_depth_to_shape(multicam_depth_m, zed_depth_m.shape[:2])

        min_depth = float(self.get_parameter("min_eval_depth_m").value)
        max_depth = float(self.get_parameter("max_eval_depth_m").value)
        valid_ref = np.isfinite(zed_depth_m) & (zed_depth_m >= min_depth) & (zed_depth_m <= max_depth)
        valid_mono = valid_ref & np.isfinite(mono_eval) & (mono_eval > 0.0)
        valid_multicam = valid_ref & np.isfinite(multicam_eval) & (multicam_eval > 0.0)

        self.log_supervised_metrics("mono", mono_eval, zed_depth_m, valid_mono)
        self.log_supervised_metrics("multicam", multicam_eval, zed_depth_m, valid_multicam)

    def log_supervised_metrics(
        self,
        label: str,
        pred_m: np.ndarray,
        ref_m: np.ndarray,
        mask: np.ndarray,
    ) -> None:
        if not mask.any():
            self.get_logger().warn(f"DA3 {label} vs ZED: no valid overlapping depth pixels")
            return

        raw = supervised_depth_metrics(pred_m, ref_m, mask)
        scale = median_scale_to_reference(pred_m, ref_m, mask)
        scaled = supervised_depth_metrics(pred_m * scale, ref_m, mask)
        self.get_logger().info(
            f"DA3 {label} vs ZED "
            f"valid={mask.mean() * 100.0:.1f}% "
            f"raw_rmse={raw['rmse']:.3f}m raw_mae={raw['mae']:.3f}m "
            f"raw_abs_rel={raw['abs_rel']:.3f} raw_delta1={raw['delta1']:.3f} "
            f"median_scale={scale:.3f} "
            f"scaled_rmse={scaled['rmse']:.3f}m scaled_abs_rel={scaled['abs_rel']:.3f}"
        )


def main() -> None:
    rclpy.init()
    node = Da3DepthMulticamNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
