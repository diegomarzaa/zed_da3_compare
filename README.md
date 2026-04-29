# zed_da3_compare

Paquete ROS 2 para comparar profundidad de Depth Anything 3 con ZED. Publica profundidad DA3 a partir de un topic de imagen con diferentes utilidades y opciones de procesamiento. La ZED se lanza aparte con el wrapper oficial.

## Nodo incluido

```text
zed_da3_compare/da3_depth_node.py
```

Hace solo esto:

```text
sensor_msgs/Image RGB/BGR/RGBA/BGRA  ->  sensor_msgs/Image 32FC1 depth
```

Topic por defecto de entrada:

```text
/zed/zed_node/rgb/color/rect/image
```

Topic por defecto de salida:

```text
/da3/depth/image
```

La salida se publica como `32FC1`, en metros según la escala que devuelva el modelo DA3 usado.

Además se publica un topic auxiliar para inspección visual:

```text
/da3/depth/preview
```

Ese preview ya va normalizado y coloreado. No es el dato científico, solo una ayuda para los ojos.

## Modelo DA3

El nodo solo acepta modelo local. Pasar por .env o por launch. Asumimos la variable de entorno:

```bash
DA3_MODEL_DIR=/home/usuario/depth_anything_ws/src/Depth-Anything-3/da3_streaming/weights/DA3NESTED-GIANT-LARGE-1.1
```

## Uso mínimo

Terminal 1, ZED oficial:

```bash
ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zed
```

Terminal 2, DA3:

```bash
ros2 run zed_da3_compare da3_depth_node.py --ros-args \
  -p model_dir:=$DA3_MODEL_DIR \
  -p process_res:=504 \
  -p process_res_method:=upper_bound_resize \
  -p process_every_n:=1
```

## Parámetros

```bash
ros2 run zed_da3_compare da3_depth_node.py --ros-args --print-params
```

```text
input_image_topic    default: /zed/zed_node/rgb/color/rect/image
output_depth_topic   default: /da3/depth/image
output_preview_topic  default: /da3/depth/preview
model_dir            obligatorio; carpeta local del modelo DA3
device               default: cuda
process_res          default: 504
process_res_method   default: upper_bound_resize
process_every_n      default: 1
match_input_size     default: true
preview_low_percentile    default: 1.0
preview_high_percentile   default: 99.0
preview_use_inverse_depth default: true
```

`match_input_size:=true` reescala la profundidad DA3 al tamaño de la imagen recibida. Así, con tu ZED actual, la salida queda en `640x360`, igual que `/zed/zed_node/depth/depth_registered`.

`/da3/depth/preview` no debe usarse para comparar métricamente. Está normalizado por percentiles sobre inversa de profundidad para que la escena se lea bien visualmente.

## Comprobaciones

```bash
ros2 topic echo /zed/zed_node/rgb/color/rect/image --once | grep encoding
ros2 topic echo /zed/zed_node/depth/depth_registered --once | grep encoding
ros2 topic echo /da3/depth/image --once | grep encoding
```

Para ver valores reales de depth, no mires `data` con `ros2 topic echo`, porque son bytes crudos. Usa un pequeño script Python o un nodo comparador más adelante.
