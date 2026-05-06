#!/usr/bin/env python3
"""Subscribe to a ROS image topic and publish Depth Anything 3 depth as 32FC1."""

from __future__ import annotations

import cv2
import numpy as np
import rclpy
import torch
from rclpy.node import Node
from sensor_msgs.msg import Image

from zed_da3_compare.da3_common import load_da3_model, log_depth_stats
from zed_da3_compare.ros_image_utils import depth_to_image_msg, depth_to_preview_msg, image_msg_to_rgb


class Da3DepthNode(Node):
    def __init__(self) -> None:
        super().__init__("da3_depth_node")

        self.declare_parameter("input_image_topic", "/zed/zed_node/rgb/color/rect/image")
        self.declare_parameter("output_depth_topic", "/da3/depth/image")
        self.declare_parameter("model_dir", "")
        self.declare_parameter("device", "cuda")
        self.declare_parameter("process_res", 504)
        self.declare_parameter("process_res_method", "upper_bound_resize")
        self.declare_parameter("process_every_n", 1)
        self.declare_parameter("match_input_size", True)
        self.declare_parameter("output_preview_topic", "/da3/depth/preview")
        self.declare_parameter("preview_low_percentile", 1.0)
        self.declare_parameter("preview_high_percentile", 99.0)
        self.declare_parameter("preview_use_inverse_depth", True)

        self.device = str(self.get_parameter("device").value)
        self.model = load_da3_model(self)
        self.frame_count = 0

        input_topic = str(self.get_parameter("input_image_topic").value)
        output_topic = str(self.get_parameter("output_depth_topic").value)
        preview_topic = str(self.get_parameter("output_preview_topic").value)

        self.depth_pub = self.create_publisher(Image, output_topic, 1)
        self.preview_pub = self.create_publisher(Image, preview_topic, 1)
        self.create_subscription(Image, input_topic, self.on_image, 1)

        self.get_logger().info(f"Subscribed: {input_topic}")
        self.get_logger().info(f"Publishing:  {output_topic} [sensor_msgs/Image, 32FC1, metres]")
        self.get_logger().info(f"Publishing:  {preview_topic} [sensor_msgs/Image, rgb8, diagnostic preview]")

    def on_image(self, msg: Image) -> None:
        # Este callback se ejecuta por cada imagen recibida del topic de entrada.
        # La idea aquí es mantener el trabajo por frame lo más corto posible:
        # 1) decidir si este frame se procesa o se salta,
        # 2) convertir la imagen ROS a numpy,
        # 3) ejecutar DA3,
        # 4) publicar el mapa bruto 32FC1,
        # 5) publicar una vista previa coloreada para inspección humana.
        self.frame_count += 1
        process_every_n = int(self.get_parameter("process_every_n").value)
        if self.frame_count % max(1, process_every_n) != 0:
            # Si se quiere bajar carga, podemos procesar solo 1 de cada N frames.
            # Esto no cambia el comportamiento del modelo; solo reduce frecuencia.
            return

        # Convertimos el sensor_msgs/Image de ROS a un array RGB HxWx3 uint8.
        # El modelo DA3 trabaja sobre este formato de entrada.
        rgb = image_msg_to_rgb(msg)
        process_res = int(self.get_parameter("process_res").value)
        process_res_method = str(self.get_parameter("process_res_method").value)

        # Inferencia sin gradientes:
        # - torch.inference_mode() evita construir grafo,
        # - la llamada a self.model.inference() devuelve la predicción de profundidad.
        with torch.inference_mode():
            prediction = self.model.inference(
                image=[rgb],
                process_res=process_res,
                process_res_method=process_res_method,
                export_dir=None,
            )

        # prediction.depth[0] es el mapa de profundidad del primer frame.
        # Lo convertimos a float32 por compatibilidad con ROS y para tener
        # un buffer compacto y predecible en memoria.
        depth_m = np.asarray(prediction.depth[0], dtype=np.float32)

        # Estadísticas de depuración:
        # esto no modifica la salida, solo nos ayuda a ver si el rango es razonable.
        # Si el modelo está devolviendo valores absurdos, aquí se ve enseguida.
        if self.frame_count % 30 == 0:
            log_depth_stats(self.get_logger(), "DA3 mono", prediction, depth_m)

        # Si el output del modelo viene a una resolución distinta de la de entrada,
        # lo reescalamos para que el topic bruto tenga el mismo tamaño que la imagen original.
        # Esto no cambia el tipo de dato, solo la rejilla espacial.
        if bool(self.get_parameter("match_input_size").value):
            depth_m = cv2.resize(depth_m, (msg.width, msg.height), interpolation=cv2.INTER_LINEAR)

        # Publicación principal:
        # 32FC1 es el dato que queremos usar para comparación, registro o métricas.
        self.depth_pub.publish(depth_to_image_msg(depth_m, msg))

        # Publicación secundaria:
        # esta imagen ya viene coloreada y normalizada para que el ojo humano
        # pueda interpretar estructura y contraste sin pelearse con el rango crudo.
        self.preview_pub.publish(
            depth_to_preview_msg(
                depth_m,
                msg,
                low_percentile=float(self.get_parameter("preview_low_percentile").value),
                high_percentile=float(self.get_parameter("preview_high_percentile").value),
                use_inverse_depth=bool(self.get_parameter("preview_use_inverse_depth").value),
            )
        )


def main() -> None:
    rclpy.init()
    node = Da3DepthNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
