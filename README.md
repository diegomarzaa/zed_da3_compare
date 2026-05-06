# zed_da3_compare

Paquete ROS 2 para comparar profundidad de Depth Anything 3 con ZED. Publica profundidad DA3 a partir de un topic de imagen con diferentes utilidades y opciones de procesamiento. La ZED se lanza aparte con el wrapper oficial.

## Nodo incluido

```text
zed_da3_compare/da3_depth_node.py
zed_da3_compare/da3_depth_multicam_node.py
zed_da3_compare/da3_offline_bag_eval.py
zed_da3_compare/da3_stereo_depth_node.py
zed_da3_compare/zed_sync_sampler_node.py
```

`da3_depth_node.py` hace esto:

```text
sensor_msgs/Image RGB/BGR/RGBA/BGRA  ->  sensor_msgs/Image 32FC1 depth
```

`da3_stereo_depth_node.py` hace esto:

```text
left Image + right Image [+ CameraInfo left/right] -> left sensor_msgs/Image 32FC1 depth
```

`da3_depth_multicam_node.py` hace esto:

```text
left Image + right Image -> mono depth + multiview depth
```

No usa `CameraInfo`, intrínsecos, extrínsecos ni baseline. Carga DA3 una sola vez y hace dos inferencias por par.

`zed_sync_sampler_node.py` hace esto:

```text
left Image + right Image + ZED depth -> left/right/depth sincronizados a pocos Hz
```

Este nodo es la forma recomendada de preparar rosbags para métricas: elige triples por `header.stamp` y publica topics nuevos ya sincronizados.

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

Para usar el nodo DA3 estéreo hace falta activar explícitamente la publicación left/right del wrapper:

```bash
ros2 launch zed_wrapper zed_camera.launch.py \
  camera_model:=zed \
  param_overrides:="video.publish_left_right:=true"
```

Antes de lanzar DA3 estéreo, comprueba que el wrapper está publicando las dos cámaras rectificadas:

```bash
ros2 topic list | grep -E "left|right|camera_info|depth_registered"
ros2 topic echo /zed/zed_node/left/color/rect/camera_info --once
ros2 topic echo /zed/zed_node/right/color/rect/camera_info --once
```

Si solo aparece un topic RGB tipo `/zed/zed_node/rgb/color/rect/image`, el nodo mono ya puede funcionar, pero el nodo estéreo necesita habilitar/publicar los topics left/right rectificados en el wrapper de la ZED.

Terminal 2, comparador:

```bash
ros2 run zed_da3_compare da3_depth_multicam_node.py --ros-args \
  -p model_dir:=$DA3_MODEL_DIR \
  -p process_res:=504 \
  -p process_res_method:=upper_bound_resize \
  -p process_every_n:=1
```

Para comparar visualmente en `rqt_image_view`, abre:

```text
/da3_compare/mono/preview
/da3_compare/multicam/preview
```

El comparador también imprime métricas en terminal cada `metrics_every_n` pares procesados:

```text
DA3 mono vs multicam ...
DA3 mono vs ZED ...
DA3 multicam vs ZED ...
```

La comparación contra ZED usa `/zed/zed_node/depth/depth_registered` como referencia práctica, no como ground truth perfecto.

## Workflow offline con rosbags

Para sacar comparativas reproducibles, usa rosbags en vez del nodo online. Los bags se guardan en:

```text
/home/usuario/depth_anything_ws/src/zed_da3_compare/data
```

Los resultados se escriben en:

```text
/home/usuario/depth_anything_ws/src/zed_da3_compare/results
```

### 1) Grabar bag a pocos Hz con sincronización correcta

Para métricas, no conviene hacer `topic_tools throttle` por separado sobre left, right y depth. Cada throttle decide independientemente qué mensajes deja pasar; aunque todos vayan a 2 Hz, puedes acabar grabando RGB de un instante y depth de otro. Para inspección visual rápida puede valer, pero para comparar errores es mejor publicar triples sincronizados.

Terminal 1, lanza la ZED con left/right:

```bash
ros2 launch zed_wrapper zed_camera.launch.py \
  camera_model:=zed \
  param_overrides:="video.publish_left_right:=true"
```

Terminal 2, publica una versión sincronizada y reducida a 2 Hz:

```bash
cd /home/usuario/depth_anything_ws
source install/setup.bash

ros2 run zed_da3_compare zed_sync_sampler_node.py --ros-args \
  -p sample_rate_hz:=2.0 \
  -p sync_tolerance_ms:=20.0
```

Topics generados:

```text
/zed_da3_eval/left/color/rect/image
/zed_da3_eval/right/color/rect/image
/zed_da3_eval/depth/depth_registered
/zed_da3_eval/left/color/rect/camera_info
/zed_da3_eval/right/color/rect/camera_info
```

Comprueba que salen a 2 Hz:

```bash
ros2 topic hz /zed_da3_eval/left/color/rect/image
ros2 topic hz /zed_da3_eval/right/color/rect/image
ros2 topic hz /zed_da3_eval/depth/depth_registered
```

Terminal 3, graba esos topics nuevos con compresión:

```bash
cd /home/usuario/depth_anything_ws/src/zed_da3_compare/data

ros2 bag record -o zed_eval_01_sampled_2hz \
  --compression-mode file \
  --compression-format zstd \
  /zed_da3_eval/left/color/rect/image \
  /zed_da3_eval/right/color/rect/image \
  /zed_da3_eval/depth/depth_registered \
  /zed_da3_eval/left/color/rect/camera_info \
  /zed_da3_eval/right/color/rect/camera_info
```

Si tu instalación de `ros2 bag` no acepta `zstd`, usa el mismo comando sin las dos opciones de compresión.

El evaluador offline descomprime automáticamente bags `zstd` en:

```text
/home/usuario/depth_anything_ws/src/zed_da3_compare/results/_bag_cache
```

El bag original de `data/` no se modifica.

Para revisar qué has grabado:

```bash
ros2 bag info /home/usuario/depth_anything_ws/src/zed_da3_compare/data/zed_eval_01_sampled_2hz
```

### 2) Alternativa rápida con topic_tools throttle

Solo la usaría para bags ligeros de inspección visual, no para métricas finales:

```bash
ros2 run topic_tools throttle messages /zed/zed_node/left/color/rect/image 2.0 /zed_throttle/left/color/rect/image
ros2 run topic_tools throttle messages /zed/zed_node/right/color/rect/image 2.0 /zed_throttle/right/color/rect/image
ros2 run topic_tools throttle messages /zed/zed_node/depth/depth_registered 2.0 /zed_throttle/depth/depth_registered
```

El problema es que estos tres procesos no comparten decisión de muestreo. Para `rqt_image_view` está bien; para `raw_rmse`, `abs_rel`, `delta1`, etc. usa `zed_sync_sampler_node.py`.

### 3) Grabar bag completo sin muestreo

Con la ZED publicando left/right y depth registrado:

```bash
cd /home/usuario/depth_anything_ws/src/zed_da3_compare/data

ros2 bag record -o zed_eval_01 \
  /zed/zed_node/left/color/rect/image \
  /zed/zed_node/right/color/rect/image \
  /zed/zed_node/depth/depth_registered \
  /zed/zed_node/left/color/rect/camera_info \
  /zed/zed_node/right/color/rect/camera_info
```

Para esta rama 1.1 vs 1.2, el evaluador usa solo:

```text
/zed/zed_node/left/color/rect/image
/zed/zed_node/right/color/rect/image
/zed/zed_node/depth/depth_registered
```

Los `CameraInfo` quedan grabados para futuras ramas con intrínsecos/extrínsecos.

### 4) Evaluación rápida

Después de compilar el paquete:

```bash
cd /home/usuario/depth_anything_ws
colcon build --packages-select zed_da3_compare --symlink-install
source install/setup.bash
```

Ejecuta primero pocos frames para comprobar sincronización, memoria y outputs:

```bash
DA3_LOG_LEVEL=WARN ros2 run zed_da3_compare da3_offline_bag_eval.py \
  --bag zed_eval_01_sampled_2hz \
  --model-dir "$DA3_MODEL_DIR" \
  --left-topic /zed_da3_eval/left/color/rect/image \
  --right-topic /zed_da3_eval/right/color/rect/image \
  --zed-depth-topic /zed_da3_eval/depth/depth_registered \
  --sync-tolerance-ms 20 \
  --max-frames 20 \
  --save-visual-every 1
```

Si evalúas un bag grabado con los topics originales, no pases `--left-topic`, `--right-topic` ni `--zed-depth-topic`, porque esos son los defaults. Si no encuentra triples sincronizados, aumenta tolerancia:

```bash
DA3_LOG_LEVEL=WARN ros2 run zed_da3_compare da3_offline_bag_eval.py \
  --bag zed_eval_01 \
  --model-dir "$DA3_MODEL_DIR" \
  --max-frames 20 \
  --sync-tolerance-ms 80 \
  --save-visual-every 1
```

Si detectas un desfase fijo entre RGB y depth, puedes compensarlo. Por ejemplo, si el depth va 800 ms tarde:

```bash
DA3_LOG_LEVEL=WARN ros2 run zed_da3_compare da3_offline_bag_eval.py \
  --bag zed_eval_01 \
  --model-dir "$DA3_MODEL_DIR" \
  --max-frames 20 \
  --zed-depth-time-offset-ms -800
```

### 5) Evaluación completa

```bash
DA3_LOG_LEVEL=WARN ros2 run zed_da3_compare da3_offline_bag_eval.py \
  --bag zed_eval_01_sampled_2hz \
  --model-dir "$DA3_MODEL_DIR" \
  --run-name zed_eval_01_sampled_2hz_da3nested_res504 \
  --left-topic /zed_da3_eval/left/color/rect/image \
  --right-topic /zed_da3_eval/right/color/rect/image \
  --zed-depth-topic /zed_da3_eval/depth/depth_registered \
  --process-res 504 \
  --process-res-method upper_bound_resize \
  --sync-tolerance-ms 20 \
  --save-visual-every 25
```

Para evaluar más rápido, procesa uno de cada N triples:

```bash
DA3_LOG_LEVEL=WARN ros2 run zed_da3_compare da3_offline_bag_eval.py \
  --bag zed_eval_01 \
  --model-dir "$DA3_MODEL_DIR" \
  --stride 5 \
  --save-visual-every 10
```

### 6) Exportar un bag reutilizable

Si quieres un rosbag para llevártelo a casa, lanza la evaluación con exportación activada. El bag resultante incluye:

```text
/zed_da3_eval/left/color/rect/image
/zed_da3_eval/right/color/rect/image
/zed_da3_eval/depth/depth_registered
/da3_compare/mono/depth/image
/da3_compare/mono/preview
/da3_compare/multicam/depth/image
/da3_compare/multicam/preview
```

Comando:

```bash
DA3_LOG_LEVEL=WARN ros2 run zed_da3_compare da3_offline_bag_eval.py \
  --bag zed_eval_01_sampled_2hz \
  --model-dir "$DA3_MODEL_DIR" \
  --run-name zed_eval_01_sampled_2hz_da3nested_res504_replay \
  --left-topic /zed_da3_eval/left/color/rect/image \
  --right-topic /zed_da3_eval/right/color/rect/image \
  --zed-depth-topic /zed_da3_eval/depth/depth_registered \
  --process-res 504 \
  --process-res-method upper_bound_resize \
  --sync-tolerance-ms 20 \
  --export-bag
```

El bag exportado queda dentro de la carpeta del run en `results/`, como `replay_bag/`. Si ya existe, el exportador usa un sufijo automático tipo `replay_bag_01`, `replay_bag_02`, etc.

### 7) Outputs

Cada run crea una carpeta en `results/`:

```text
results/<run_name>/
  metadata.json
  summary.json
  metrics_per_frame.csv
  arrays/
    zed/000000.npy
    mono/000000.npy
    multicam/000000.npy
  visuals/
    000000_left.png
    000000_right.png
    000000_zed_depth.png
    000000_mono_depth.png
    000000_multicam_depth.png
    000000_mono_abs_error.png
    000000_multicam_abs_error.png
    000000_mono_multicam_abs_diff.png
  plots/
    abs_rel_over_frames.png
    rmse_over_frames.png
    pairwise_over_frames.png
    scale_over_frames.png
    scaled_abs_rel_hist.png
```

`metrics_per_frame.csv` es el archivo principal para análisis externo. `summary.json` contiene medias, medianas, desviaciones, mínimos, máximos y conteos.

### 8) Métricas offline

Supervisadas contra ZED:

```text
raw_rmse, raw_mae, raw_abs_rel, raw_sq_rel
raw_rmse_log, raw_log10, raw_silog
raw_delta1, raw_delta2, raw_delta3
scaled_* tras ajustar escala mediana contra ZED
```

No supervisadas entre mono y multiview:

```text
pair_mae
pair_rmse
pair_mean_rel_to_a
pair_symmetric_rel
pair_median_scale_b_over_a
pair_corr
pair_grad_mae
```

Columnas `winner_*`:

```text
winner_raw_abs_rel
winner_scaled_abs_rel
winner_raw_rmse
winner_scaled_rmse
```

Estas indican qué rama ganó en cada frame para esa métrica.

La lectura recomendada es:

```text
raw_*      mide escala absoluta DA3 contra ZED
scaled_*   mide forma relativa tras corregir escala global por frame
pair_*     mide cuánto cambia DA3 al darle right además de left
plots/*    permite ver si una rama gana siempre o solo en escenas concretas
visuals/*  sirve para revisar casos malos frame a frame
```

## Parámetros

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

Parámetros principales de `da3_depth_multicam_node.py`:

```text
left_image_topic               default: /zed/zed_node/left/color/rect/image
right_image_topic              default: /zed/zed_node/right/color/rect/image
mono_output_depth_topic        default: /da3_compare/mono/depth/image
mono_output_preview_topic      default: /da3_compare/mono/preview
multicam_output_depth_topic    default: /da3_compare/multicam/depth/image
multicam_output_preview_topic  default: /da3_compare/multicam/preview
model_dir                      obligatorio; carpeta local del modelo DA3
device                         default: cuda
process_res                    default: 504
process_res_method             default: upper_bound_resize
process_every_n                default: 1
match_input_size               default: true
max_stamp_delta_ms             default: 40.0
zed_depth_topic                default: /zed/zed_node/depth/depth_registered
enable_metrics                 default: true
metrics_every_n                default: 10
max_depth_stamp_delta_ms       default: 80.0
min_eval_depth_m               default: 0.2
max_eval_depth_m               default: 20.0
ref_view_strategy              default: saddle_balanced
```

`match_input_size:=true` reescala la profundidad DA3 al tamaño de la imagen recibida. Así, con tu ZED actual, la salida queda en `640x360`, igual que `/zed/zed_node/depth/depth_registered`.

`/da3/depth/preview` no debe usarse para comparar métricamente. Está normalizado por percentiles sobre inversa de profundidad para que la escena se lea bien visualmente.

`/da3_compare/mono/depth/image` es la rama 1.1: DA3 recibe solo la imagen izquierda.

`/da3_compare/multicam/depth/image` es la rama 1.2: DA3 recibe dos imágenes RGB y estima profundidad multivista sin información externa de cámara.

Métricas impresas:

```text
mono vs multicam:
  mae                     diferencia media absoluta entre ambas salidas DA3
  mean_rel                diferencia relativa respecto a mono
  median_scale_multi_over_mono  escala mediana multicam/mono
  corr                    correlación entre ambos mapas de profundidad

mono/multicam vs ZED:
  raw_rmse, raw_mae       error métrico directo contra ZED
  raw_abs_rel             error relativo absoluto directo
  raw_delta1              porcentaje con ratio < 1.25
  median_scale            escala mediana para ajustar DA3 a ZED
  scaled_rmse, scaled_abs_rel  error tras aplicar esa escala
```

## Comprobaciones

```bash
ros2 topic echo /zed/zed_node/rgb/color/rect/image --once | grep encoding
ros2 topic echo /zed/zed_node/depth/depth_registered --once | grep encoding
ros2 topic echo /da3_compare/mono/depth/image --once | grep encoding
ros2 topic echo /da3_compare/multicam/depth/image --once | grep encoding
```

Para ver valores reales de depth, no mires `data` con `ros2 topic echo`, porque son bytes crudos. Usa un pequeño script Python o un nodo comparador más adelante.
