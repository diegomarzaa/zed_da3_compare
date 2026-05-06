#!/usr/bin/env python3
"""Publish pose-conditioned DA3 depth from a rectified ZED stereo pair."""

from __future__ import annotations

import cv2
import numpy as np
import rclpy
import torch
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image

from zed_da3_compare.da3_common import (
    baseline_from_right_camera_info,
    camera_info_to_intrinsics,
    load_da3_model,
    log_depth_stats,
    stamp_to_seconds,
    stereo_extrinsics_from_baseline,
)
from zed_da3_compare.ros_image_utils import depth_to_image_msg, depth_to_preview_msg, image_msg_to_rgb


class Da3StereoDepthNode(Node):
    def __init__(self) -> None:
        super().__init__("da3_stereo_depth_node")

        self.declare_parameter("left_image_topic", "/zed/zed_node/left/color/rect/image")
        self.declare_parameter("right_image_topic", "/zed/zed_node/right/color/rect/image")
        self.declare_parameter("left_camera_info_topic", "/zed/zed_node/left/color/rect/camera_info")
        self.declare_parameter("right_camera_info_topic", "/zed/zed_node/right/color/rect/camera_info")
        self.declare_parameter("output_depth_topic", "/da3_stereo/depth/left/image")
        self.declare_parameter("output_preview_topic", "/da3_stereo/depth/left/preview")
        self.declare_parameter("model_dir", "")
        self.declare_parameter("device", "cuda")
        self.declare_parameter("process_res", 504)
        self.declare_parameter("process_res_method", "upper_bound_resize")
        self.declare_parameter("process_every_n", 1)
        self.declare_parameter("match_input_size", True)
        self.declare_parameter("max_stamp_delta_ms", 40.0)
        self.declare_parameter("use_camera_geometry", True)
        self.declare_parameter("baseline_m", 0.0)
        self.declare_parameter("align_to_input_ext_scale", True)
        self.declare_parameter("use_ray_pose", False)
        self.declare_parameter("ref_view_strategy", "saddle_balanced")
        self.declare_parameter("preview_low_percentile", 1.0)
        self.declare_parameter("preview_high_percentile", 99.0)
        self.declare_parameter("preview_use_inverse_depth", True)

        self.device = str(self.get_parameter("device").value)
        self.model = load_da3_model(self)

        self.left_msg: Image | None = None
        self.right_msg: Image | None = None
        self.left_info: CameraInfo | None = None
        self.right_info: CameraInfo | None = None
        self.frame_count = 0
        self.last_pair_key: tuple[int, int, int, int] | None = None
        self.warned_waiting_for_info = False
        self.warned_missing_baseline = False

        left_image_topic = str(self.get_parameter("left_image_topic").value)
        right_image_topic = str(self.get_parameter("right_image_topic").value)
        left_info_topic = str(self.get_parameter("left_camera_info_topic").value)
        right_info_topic = str(self.get_parameter("right_camera_info_topic").value)
        output_topic = str(self.get_parameter("output_depth_topic").value)
        preview_topic = str(self.get_parameter("output_preview_topic").value)

        self.depth_pub = self.create_publisher(Image, output_topic, 1)
        self.preview_pub = self.create_publisher(Image, preview_topic, 1)
        self.create_subscription(Image, left_image_topic, self.on_left_image, 1)
        self.create_subscription(Image, right_image_topic, self.on_right_image, 1)
        self.create_subscription(CameraInfo, left_info_topic, self.on_left_info, 1)
        self.create_subscription(CameraInfo, right_info_topic, self.on_right_info, 1)

        self.get_logger().info(f"Subscribed left image:  {left_image_topic}")
        self.get_logger().info(f"Subscribed right image: {right_image_topic}")
        self.get_logger().info(f"Subscribed left info:   {left_info_topic}")
        self.get_logger().info(f"Subscribed right info:  {right_info_topic}")
        self.get_logger().info(f"Publishing: {output_topic} [sensor_msgs/Image, 32FC1, left depth]")
        self.get_logger().info(f"Publishing: {preview_topic} [sensor_msgs/Image, rgb8, diagnostic preview]")

    def on_left_info(self, msg: CameraInfo) -> None:
        self.left_info = msg
        self.try_process_pair()

    def on_right_info(self, msg: CameraInfo) -> None:
        self.right_info = msg
        self.try_process_pair()

    def on_left_image(self, msg: Image) -> None:
        self.left_msg = msg
        self.try_process_pair()

    def on_right_image(self, msg: Image) -> None:
        self.right_msg = msg
        self.try_process_pair()

    def try_process_pair(self) -> None:
        # El nodo procesa pares left/right. Como evitamos meter message_filters
        # de momento, guardamos el último mensaje de cada cámara y comprobamos
        # que sus stamps estén suficientemente cerca antes de lanzar DA3.
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

        # C1/C2 se deciden con use_camera_geometry:
        # - False: DA3 recibe dos imágenes y estima consistencia multivista sin calibración explícita.
        # - True: además recibe intrínsecos y extrínsecos reales de la ZED rectificada.
        intrinsics, extrinsics = self.build_camera_geometry()
        if bool(self.get_parameter("use_camera_geometry").value) and (intrinsics is None or extrinsics is None):
            return

        left_rgb = image_msg_to_rgb(self.left_msg)
        right_rgb = image_msg_to_rgb(self.right_msg)
        process_res = int(self.get_parameter("process_res").value)
        process_res_method = str(self.get_parameter("process_res_method").value)

        with torch.inference_mode():
            prediction = self.model.inference(
                image=[left_rgb, right_rgb],
                extrinsics=extrinsics,
                intrinsics=intrinsics,
                align_to_input_ext_scale=bool(self.get_parameter("align_to_input_ext_scale").value),
                use_ray_pose=bool(self.get_parameter("use_ray_pose").value),
                ref_view_strategy=str(self.get_parameter("ref_view_strategy").value),
                process_res=process_res,
                process_res_method=process_res_method,
                export_dir=None,
            )

        # Publicamos la profundidad asociada a la vista izquierda para poder
        # compararla directamente con /zed/.../depth_registered.
        left_depth_m = np.asarray(prediction.depth[0], dtype=np.float32)
        if self.frame_count % 30 == 0:
            log_depth_stats(self.get_logger(), "DA3 stereo left", prediction, left_depth_m)

        if bool(self.get_parameter("match_input_size").value):
            left_depth_m = cv2.resize(
                left_depth_m,
                (self.left_msg.width, self.left_msg.height),
                interpolation=cv2.INTER_LINEAR,
            )

        self.depth_pub.publish(depth_to_image_msg(left_depth_m, self.left_msg))
        self.preview_pub.publish(
            depth_to_preview_msg(
                left_depth_m,
                self.left_msg,
                low_percentile=float(self.get_parameter("preview_low_percentile").value),
                high_percentile=float(self.get_parameter("preview_high_percentile").value),
                use_inverse_depth=bool(self.get_parameter("preview_use_inverse_depth").value),
            )
        )

    def build_camera_geometry(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        if not bool(self.get_parameter("use_camera_geometry").value):
            return None, None

        if self.left_info is None or self.right_info is None:
            if not self.warned_waiting_for_info:
                self.get_logger().warn("Waiting for left/right CameraInfo before pose-conditioned DA3 inference")
                self.warned_waiting_for_info = True
            return None, None

        baseline_m = float(self.get_parameter("baseline_m").value)
        if baseline_m <= 0.0:
            inferred = baseline_from_right_camera_info(self.right_info)
            if inferred is not None:
                baseline_m = inferred

        if baseline_m <= 0.0:
            if not self.warned_missing_baseline:
                self.get_logger().warn(
                    "Could not infer stereo baseline from right CameraInfo.P. "
                    "Set baseline_m explicitly or disable use_camera_geometry."
                )
                self.warned_missing_baseline = True
            return None, None

        intrinsics = np.stack(
            [
                camera_info_to_intrinsics(self.left_info),
                camera_info_to_intrinsics(self.right_info),
            ],
            axis=0,
        ).astype(np.float32)
        extrinsics = stereo_extrinsics_from_baseline(baseline_m)
        return intrinsics, extrinsics


def main() -> None:
    rclpy.init()
    node = Da3StereoDepthNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
