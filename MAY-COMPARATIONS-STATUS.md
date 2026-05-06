# MAY - Comparaciones ZED / DA3

## Objetivo principal ahora

Quiero centrarme en comparaciones experimentales entre depth ZED y depth DA3, pero sin depender tanto de ground truth fisico.

El ground truth fisico consume mucho tiempo, es incomodo de medir y, en la practica, tampoco estoy obteniendo una verdad perfecta. Muchas medidas son aproximadas: hasta un punto del objeto, con cinta metrica, con ROIs no exactas, con objetos que no son planos, etc.

La prioridad actual pasa a ser:

RGB ZED -> depth ZED -> depth DA3 -> comparacion escena/ROI/pixel

La idea es obtener bastantes frames variados y estudiar como se desenvuelven ambos metodos. En este modo, la ZED se usa como referencia practica o pseudo-ground-truth. No significa que la ZED sea verdad absoluta, pero si es una referencia metrica razonable para empezar a sacar patrones.

Ground truth fisico queda como validacion puntual:

1. Para escenas pequenas y controladas.
2. Para comprobar escala absoluta.
3. Para casos donde ZED y DA3 discrepen mucho.
4. Para underwater si necesito defender un resultado concreto.

Mono vs multiview queda como pendiente, pero no como foco principal ahora. En las pruebas iniciales no parecio aportar una mejora clara fuera del agua. Puede tener sentido volver a probarlo underwater, o incluso con mas camaras, pero no quiero que eso bloquee la comparacion principal.

## Linea seguida hasta ahora

El paquete `zed_da3_compare` empezo como una integracion simple de DA3 con ROS 2 usando imagenes de la ZED. Poco a poco fue creciendo hacia un banco de evaluacion.

La linea ha sido: ZED publica RGB/depth
-> DA3 procesa RGB
-> publico depth DA3 en ROS
-> comparo DA3 mono vs DA3 multiview
-> comparo DA3 vs depth ZED
-> detecto problemas de sincronizacion
-> paso a workflow offline con rosbags
-> guardo metricas, arrays, visuales y plots

La conclusion actual es que el workflow offline es el camino mas limpio para experimentos serios, porque asegura que cada RGB, cada depth ZED y cada salida DA3 correspondan al frame correcto.

## Tareas pendientes

Prioridad alta:

1. Capturar mas frames variados con la herramienta de anotacion/captura.
2. Guardar muestras con RGB, depth ZED y depth DA3 siempre que sea posible.
3. Analizar diferencias ZED vs DA3 por imagen completa, por ROI y visualmente.
4. Generar graficas claras: scatter ZED/DA3, error DA3-ZED, histogramas, overlays.
5. Usar el visor interactivo para inspeccionar pixel a pixel donde ZED y DA3 discrepan.
6. Hacer un primer dataset sin obsesionarme con GT fisico.

Prioridad media:

1. Anotar ROIs de objetos aunque no siempre tengan GT real.
2. Usar GT fisico solo en algunas escenas concretas.
3. Separar casos por tipo: cerca/lejos, textura/poca textura, brillos, bordes, objetos finos, underwater.
4. Buscar datos/estudios sobre error esperado de la ZED para contextualizar resultados.
5. Mejorar los reportes para que comparen tambien muestras sin GT.

Prioridad baja por ahora:

1. DA3 con CameraInfo/intrinsecos/extrinsecos.
2. Wrapper ROS 2 generico y limpio.
3. Docker definitivo.
4. IMU/visual-inertial.
5. Multicamera con mas de dos camaras.

## Decision actual

Mi decision por ahora:

Priorizar resultados experimentales.
Capturar mas variedad de escenas.
Comparar DA3 contra ZED usando ZED como referencia practica.
No bloquearme con ground truth fisico.
Usar GT solo cuando aporte mucho.
Mantener workflow mixto: online para capturar y offline para analizar bien.

La pieza central ahora es:

RGB frame + depth ZED + depth DA3 + ROIs opcionales -> analisis visual y metrico

Y si hay GT:

RGB frame + ROI + distancia real -> metricas ZED/DA3 contra GT

Pero el modo principal pasa a ser:

ZED como referencia -> cuanto se separa DA3, donde se separa y por que.

## Herramientas nuevas

### Herramienta: captura/anotacion GT

Archivo: `zed_da3_compare/gt_annotation_tool.py`.

Programa OpenCV/ROS 2 para centralizar el workflow.

Idea:

ver topics ZED en vivo -> congelar frame -> seleccionar ROIs -> poner GT si existe -> guardar sample -> procesar DA3 si quiero

Puede trabajar con:

topics en vivo de la ZED
samples ya guardados
rosbags, segun el flujo que se vaya usando

Guarda por escena en:

`zed_da3_compare/captures/<scene_name>/`

Y por muestra:

`sample_000001/left.png`
`sample_000001/right.png`, si existe
`sample_000001/zed_depth.npy`
`sample_000001/da3_mono_depth.npy`, si se procesa
`sample_000001/da3_multiview_depth.npy`, si se procesa
`sample_000001/metadata.json`
`sample_000001/annotations.csv`
`sample_000001/preview_rois.png`

Tambien mantiene:

`scene_annotations.csv`

Estado:

Funcional.
Sirve tanto para anotacion online como para revisar/editar samples despues.
No tiene que ser bonito; tiene que ser practico y permitir capturar datos sin lanzar mil programas.

Actualizacion sincronizacion ZED:

La GUI usa ahora por defecto `/zed/zed_node/rgb/color/rect/image` para la imagen principal, no `/zed/zed_node/left/color/rect/image`.

Motivo:

Stereolabs documenta que `rgb` y `left` son identicos en camaras stereo, pero `rgb` es el canal pensado para asociarse al depth sincronizado.

Depth:

`/zed/zed_node/depth/depth_registered`

Right:

`/zed/zed_node/right/color/rect/image`

La derecha sigue siendo util para DA3 multiview, pero no bloquea la captura principal RGB-depth.

Cada sample guarda en `metadata.json` un bloque `verification` con:

topics usados
stamps
deltas RGB-depth/right
encodings
formas RGB/depth
si RGB y depth tienen el mismo tamano
estadisticas basicas del depth ZED

### Herramienta: analisis de escena

Archivo: `zed_da3_compare/analyze_gt_scene.py`.

Lee una carpeta de escena y genera una carpeta:

`analysis/`

Outputs:

`analysis_report.md`
`dashboard.png`
`summary_by_method.csv`
`winner_counts.csv`
`roi_metrics_wide.csv`
`roi_metrics_long.csv`
`roi_metrics_existing_wide.csv`
`roi_metrics_existing_long.csv`
`plots/*.png`
`visuals/*_roi_overlay.png`

Hace:

recalculo de metricas desde `.npy`
comparacion por ROI contra GT si existe
comparacion ZED vs DA3 si no quiero depender tanto de GT
deteccion de samples referenciados pero no presentes
graficas y overlays para entender los resultados rapido

Estado:

Funcional.
Ya se uso con `captures/scene_00_tests`.
Ya genera analisis sin GT usando ZED como referencia.

Nuevos outputs DA3-ZED:

`analysis/zed_reference_roi_metrics.csv`
`analysis/zed_reference_sample_metrics.csv`
`analysis/plots/zed_ref_roi_scatter.png`
`analysis/plots/zed_ref_roi_signed_diff.png`
`analysis/plots/zed_ref_roi_abs_diff_boxplot.png`
`analysis/plots/zed_ref_sample_mae_vs_validity.png`
`analysis/plots/zed_ref_sample_scale.png`

Lectura:

`raw_*`: DA3 contra ZED sin corregir escala.
`scaled_*`: DA3 contra ZED despues de ajustar escala global.
`diff_m`: DA3 - ZED en una ROI.
`ratio_method_over_zed`: si DA3 da mas o menos distancia que ZED.
`zed_valid_ratio`: cuidado si la ZED tiene pocos pixeles validos en esa ROI.

### Herramienta: visor interactivo de depth

Archivo: `zed_da3_compare/interactive_depth_viewer.py`.

Uso:

```bash
ros2 run zed_da3_compare interactive_depth_viewer.py \
  /home/usuario/depth_anything_ws/src/zed_da3_compare/captures/scene_00_tests
```

Permite pasar el cursor por la imagen y ver valores de profundidad.

Muestra:

ZED depth en el pixel
DA3 mono depth en el pixel
DA3 multiview depth si existe
mediana local alrededor del cursor
ROIs anotadas

Controles:

`v`: cambia vista RGB/ZED/DA3/error
`n`: siguiente sample
`p`: sample anterior
`r`: RGB
`q`: salir

Estado:

Funcional.
Muy util para comparar ZED y DA3 de forma intuitiva.
Especialmente util ahora que quiero mirar diferencias en escenas generales, no solo tablas.

### Notebook: analisis DA3-ZED paso a paso

Archivo: `notebooks/da3_zed_scene_analysis.ipynb`.

Objetivo:

entender el analisis sin tener que leer todo el script automatico.

Hace:

carga escena
carga anotaciones
calcula metricas ROI DA3-ZED
calcula metricas por sample completo
dibuja scatter, barras, escala y mapas depth
permite editar formulas y graficas rapido

Uso:

```bash
cd /home/usuario/depth_anything_ws/src/zed_da3_compare
jupyter lab notebooks/da3_zed_scene_analysis.ipynb
```

El script `analyze_gt_scene.py` queda como flujo automatico.
El notebook queda como flujo didactico/exploratorio.

## Resultados iniciales: scene_00_tests

Carpeta:

`/home/usuario/depth_anything_ws/src/zed_da3_compare/captures/scene_00_tests`

Analisis:

`/home/usuario/depth_anything_ws/src/zed_da3_compare/captures/scene_00_tests/analysis`

Reporte:

`analysis/analysis_report.md`

Estado de datos:

Solo existe realmente `sample_000003`.
`scene_annotations.csv` referencia tambien `sample_000001`, pero esa carpeta no esta.
El analizador lo marca como sample faltante/stale.
En `sample_000003` hay 6 ROIs.
Hay DA3 mono.
No hay DA3 multiview guardado.

Resultado con GT aproximado:

ZED mean abs error: 0.0968 m
ZED median abs error: 0.0639 m
DA3 mono mean abs error: 0.3668 m
DA3 mono median abs error: 0.2325 m

Ganadores por ROI:

ZED gana en 5 ROIs.
DA3 mono gana en 1 ROI.

Caso interesante:

monitor:
GT aproximado: 0.78 m
ZED: 1.08 m
DA3 mono: 0.93 m

Aqui DA3 queda mas cerca del GT, pero la ZED tenia muy pocos pixeles validos en la ROI. Esto no significa automaticamente que DA3 sea mejor en general. Puede significar que la ZED sufrio en esa zona concreta.

Lectura actual:

Con esta muestra pequena, ZED parece mucho mejor en profundidad metrica absoluta.
DA3 tiende a sobreestimar en varias ROIs.
DA3 puede ser interesante en zonas donde ZED tiene poca validez o huecos.
No puedo sacar conclusiones fuertes con una sola muestra.
La prioridad es capturar mas variedad.

## Nodos y piezas actuales

### Nodo: DA3 mono

Archivo: `zed_da3_compare/da3_depth_node.py`.

imagen ROS -> numpy RGB -> DA3 -> depth 32FC1 -> ROS topic

Publica:

/da3/depth/image: depth bruto 32FC1, para metricas.
/da3/depth/preview: imagen coloreada, solo para mirar.

Esto separa bien dato cientifico y visualizacion. El topic `32FC1` puede verse raro en un visor normal, pero eso no significa que este mal. Para mirar a ojo esta el `preview`.

Estado:

Funcional.
Util para integracion ROS 2 basica.
No es el foco principal de comparativas offline, pero sigue siendo una pieza base.

### Nodo: DA3 mono vs multiview

Archivo: `zed_da3_compare/da3_depth_multicam_node.py`.

left image + right image -> DA3 mono + DA3 multiview

Hace dos inferencias:

DA3 mono: image=[left]
DA3 multiview: image=[left, right]

Publica:

/da3_compare/mono/depth/image
/da3_compare/mono/preview
/da3_compare/multicam/depth/image
/da3_compare/multicam/preview

Tambien puede comparar contra `/zed/zed_node/depth/depth_registered`, pero las metricas online dependen mucho de la sincronizacion real de los topics.

Estado:

Funcional.
No usa CameraInfo, intrinsecos, extrinsecos ni baseline.
No se vio una mejora clara de multiview frente a mono fuera del agua.
Pendiente probar underwater y quizas con mas camaras.

### Nodo: DA3 stereo con CameraInfo

Archivo: `zed_da3_compare/da3_stereo_depth_node.py`.

left image + right image + CameraInfo -> DA3 con intrinsecos/extrinsecos

Esta rama intenta pasarle geometria real a DA3:

intrinsics desde CameraInfo.K
baseline desde right CameraInfo.P
extrinsics construidas a partir del baseline

Esto podria ser mas correcto que darle solo dos imagenes, pero tambien es mas delicado. Si las convenciones de extrinsecos, escala o baseline no coinciden con lo que espera DA3, puede dar resultados confusos.

Estado:

No es prioridad ahora.
No lo termine de probar bien.
Queda para mas adelante, cuando ya tenga clara la comparacion ZED vs DA3 en escenas variadas.

### Nodo: sampler sincronizado

Archivo: `zed_da3_compare/zed_sync_sampler_node.py`.

left + right + ZED depth -> triples sincronizados a pocos Hz

Publica:

/zed_da3_eval/left/color/rect/image
/zed_da3_eval/right/color/rect/image
/zed_da3_eval/depth/depth_registered
/zed_da3_eval/left/color/rect/camera_info
/zed_da3_eval/right/color/rect/camera_info

Por que existe:

No quiero grabar left, right y depth con throttle independiente.
Cada throttle podria quedarse con frames distintos.
Para metricas, eso seria peligroso.

El sampler busca mensajes cercanos por `header.stamp` y publica un triple ya elegido. Esto hace que el rosbag sea mas pequeno y mas fiable.

Estado:

Muy importante para workflow offline.
Seguramente es la forma recomendada de grabar datasets pequenos de comparacion.

### Evaluador offline

Archivo: `zed_da3_compare/da3_offline_bag_eval.py`.

rosbag -> sincronizar triples -> DA3 mono/multiview -> metricas -> outputs

Outputs:

metrics_per_frame.csv
summary.json
metadata.json
arrays/
visuals/
plots/

Esto es ahora la pieza central para comparar resultados de forma repetible.

Estado:

Funcional en concepto.
Pendiente resolver dependencia zstandard si el bag esta comprimido.
Para ahora no necesito obsesionarme con compresion; me vale un unico rosbag local para iterar.

## Por que offline

Me empece a decantar por hacerlo offline por sincronizacion.

En vivo puede pasar:

left RGB de un instante
right RGB de otro instante
depth ZED de otro instante
DA3 procesando con retraso variable

Aunque todo parezca ir a la vez, las metricas pueden quedar contaminadas si los stamps no cuadran.

Offline puedo hacer:

grabo pocos frames buenos
compruebo los topics
proceso siempre el mismo bag
repito con distintos parametros
guardo metricas y visuales
comparo sin depender del tiempo real

Esto tambien facilita llevarme el bag a otro ordenador, aunque ahora mismo la compresion no es prioritaria.

## Ground truth fisico

Antes lo plantee como prioridad principal. Ahora lo veo mas como una herramienta puntual.

Problema:

medir fisicamente consume mucho tiempo
la medida suele ser aproximada
la ROI puede no corresponder exactamente al punto medido
un objeto tiene volumen, inclinacion y bordes
underwater sera aun mas dificil

Por eso no quiero que el avance dependa de tener GT perfecto.

Uso actual del GT:

1. Validar algunas escenas controladas.
2. Calibrar intuicion de escala.
3. Comprobar casos donde DA3 y ZED discrepan mucho.
4. Tener algunas graficas defendibles de error absoluto real.

Ejemplo:

objeto A: botella aprox 0.50 m
objeto B: caja aprox 1.00 m
objeto C: pared/cartel aprox 2.50 m
objeto D: objeto pequeno aprox 4.00 m

Para cada objeto puedo comparar:

distancia real aproximada
distancia ZED
distancia DA3 mono
distancia DA3 multiview, si existe
error ZED
error DA3

Pero esto ya no es requisito para cada frame.

## Experimento recomendado ahora

### 1. Definir escenas

Quiero mas variedad antes que pocas escenas demasiado medidas.

Propuesta inicial:

Escena 01: interior seco, objetos cercanos.
Escena 02: interior seco, objetos a media distancia.
Escena 03: pasillo/exterior, mas profundidad.
Escena 04: objetos dificiles: negro, brillante, fino, poca textura.
Escena 05: underwater, cuando este listo.

Para cada escena apuntaria:

nombre de escena
fecha
camara usada
condiciones de luz
objetos visibles
si hay GT fisico o no
notas: brillos, transparencia, poca textura, agua, etc.

### 2. Capturar samples

Modo recomendado ahora:

usar `gt_annotation_tool.py`
capturar samples en vivo
guardar RGB/depth ZED
procesar DA3 desde la herramienta o despues
anotar ROIs si quiero estudiar objetos concretos
poner GT solo si lo tengo

Esto permite mover objetos, cambiar escena y capturar rapido.

### 3. Opcional: grabar rosbag pequeno

Si quiero reproducibilidad fuerte, puedo grabar rosbag.

Lanzar ZED con left/right:

```bash
ros2 launch zed_wrapper zed_camera.launch.py \
  camera_model:=zed \
  param_overrides:="video.publish_left_right:=true"
```

Lanzar sampler:

```bash
ros2 run zed_da3_compare zed_sync_sampler_node.py --ros-args \
  -p sample_rate_hz:=2.0 \
  -p sync_tolerance_ms:=20.0
```

Grabar:

```bash
ros2 bag record -o scene_01_objects_2hz \
  /zed_da3_eval/left/color/rect/image \
  /zed_da3_eval/right/color/rect/image \
  /zed_da3_eval/depth/depth_registered \
  /zed_da3_eval/left/color/rect/camera_info \
  /zed_da3_eval/right/color/rect/camera_info
```

### 4. Asociar ROIs a objetos

Necesito una forma de decir: en este frame, esta zona corresponde a este objeto.

Opciones:

Opcion simple: usar punto central del objeto.
Opcion mejor: seleccionar una ROI rectangular del objeto.
Opcion mas avanzada: mascara manual o segmentacion.

Yo empezaria por ROI rectangular. Es mas robusto que un solo pixel, porque un pixel puede caer en borde, ruido o hueco de depth.

Para cada ROI calcularia:

mediana depth ZED
media depth ZED
percentiles ZED
mediana depth DA3
media depth DA3
percentiles DA3
diferencia DA3-ZED
ratio DA3/ZED
error absoluto contra GT, si existe
error relativo contra GT, si existe

La mediana probablemente sera la metrica mas estable para objetos.

### 5. Tabla de anotaciones

La herramienta ya guarda `annotations.csv` por sample y `scene_annotations.csv` por escena.

Columnas posibles:

scene
frame_idx
object_id
object_name
gt_distance_m
x1
y1
x2
y2
notes

Ejemplo:

scene_01,000012,box_1,caja azul,1.20,350,220,470,360,textura buena
scene_01,000012,bottle_1,botella,0.75,120,250,180,410,brillo/transparencia

Si no hay GT, `gt_distance_m` puede quedar vacio o `nan`.

Esto permite comparar por objeto, no solo por frame entero.

### 6. Procesar DA3

Opciones:

procesar DA3 en la interfaz
procesar DA3 despues sobre samples guardados
procesar DA3 desde rosbag con el evaluador offline

Ahora prefiero procesar al final cuando el dataset este claro, pero que la interfaz permita lanzarlo si conviene.

### 7. Sacar resultados utiles

Outputs que quiero:

tabla por ROI: ZED vs DA3
tabla por sample: diferencias medias y medianas
histograma DA3-ZED
scatter ZED vs DA3
mapas de error DA3-ZED
visual con ROI dibujada encima de RGB/depth
ranking de casos donde DA3 se separa mucho de ZED
ranking de casos donde ZED tiene pocos validos
si hay GT: grafica error absoluto vs distancia

Esto ya daria conclusiones claras:


DA3 se parece a ZED en X.
DA3 se separa de ZED en Y.
ZED parece fallar o quedarse sin validos en Z.
DA3 tiene buena estructura pero escala desviada en Z.
Underwater cambia tal cosa.

## Metricas que me interesan

Para imagen completa:


raw_rmse
raw_mae
raw_abs_rel
scaled_rmse
scaled_abs_rel
delta1/delta2/delta3
valid_ratio_zed
valid_ratio_da3
median_diff_da3_minus_zed
median_abs_diff_da3_zed

Para ROIs sin GT:


zed_median_depth_m
da3_median_depth_m
da3_minus_zed_m
abs_da3_minus_zed_m
da3_over_zed_ratio
zed_valid_ratio
da3_valid_ratio
roi_area_px

Para ROIs con ground truth:

zed_median_depth_m
da3_median_depth_m
zed_abs_error_m
da3_abs_error_m
zed_rel_error
da3_rel_error
winner_abs_error

Interpretacion:

raw_*: DA3 contra ZED sin tocar escala.
scaled_*: forma relativa tras reescalar DA3 contra ZED.
ROI median: distancia robusta de un objeto concreto.
valid_ratio: cuanta parte de la ROI tiene depth util.
DA3-ZED: donde DA3 se separa de la referencia ZED.

Ahora mismo me interesan especialmente:

1. DA3-ZED por ROI.
2. DA3-ZED por imagen completa.
3. Casos donde ZED tiene baja validez.
4. Casos donde DA3 mantiene estructura aunque no escala.
5. Casos puntuales con GT fisico.

## Sobre ZED como referencia

Puedo usar ZED como referencia practica, pero no como verdad absoluta.

Este cambio es importante.

Sin GT fisico, la pregunta no es:

"quien tiene razon?"

La pregunta pasa a ser:

"cuanto se parece DA3 a la profundidad metrica de la ZED?"
"en que escenas se separa?"
"esa separacion parece fallo de DA3, fallo de ZED o una diferencia esperable?"

Seria util buscar o recopilar informacion sobre errores esperables de la ZED concreta:


rango recomendado de trabajo
error estimado segun distancia
problemas con poca textura
problemas con brillos/transparencias
problemas underwater
efecto de calibracion

Eso serviria para contextualizar resultados:


Si ZED falla, no significa automaticamente que DA3 este mal.
Si DA3 se acerca mas al GT que ZED en algun caso, es interesante.
Si ambos fallan, puede ser escena dificil o GT/anotacion mala.
Si DA3 se separa de ZED, necesito mirar la imagen y no solo la tabla.

Por eso el visor interactivo es importante: permite ver el valor ZED/DA3 pixel a pixel y entender si el error sale de bordes, huecos, objetos finos, reflejos o escala global.

## CameraInfo y calibracion

CameraInfo entra en dos sitios:


comparacion geometrica DA3 stereo
confianza en la propia ZED

Para el objetivo actual, no necesito resolverlo ya. Pero si quiero hacer comparaciones serias, tarde o temprano importa saber:


si la ZED esta bien calibrada
que intrinsecos K esta publicando
que baseline se infiere de CameraInfo.P
si el depth_registered esta alineado con la imagen left

Por ahora:


No priorizar CameraInfo.
Guardar CameraInfo en los rosbags igualmente.
Volver a esto si entro en DA3 con geometria explicita.

## Multicamera DA3

Estado actual:


Probado fuera del agua.
No vi gran diferencia entre mono y multiview.
No lo descartaria del todo.

Pendiente:


probar underwater
probar escenas donde una vista tenga oclusiones
probar mas camaras si algun dia hay setup disponible
comparar si mejora forma relativa aunque no mejore escala absoluta

No quiero centrarme ahora en esto porque el objetivo principal es ZED vs DA3 con muchas escenas. Multiview queda como variante a probar despues, especialmente underwater.

## DA3 ROS 2 organized wrapper

Idea: tener un paquete ROS 2 limpio, modular y reutilizable para DA3.

Algo tipo el repo externo de DA3 ROS2 Wrapper, pero menos lioso y menos sobrecomplicado.

Lo que deberia tener:


loader de modelo DA3
conversion ROS image <-> numpy
publicacion depth bruto
publicacion preview
parametros claros
nodos separados para mono/multiview/stereo
launch files limpios
README corto y operativo

Esto ya se esta haciendo parcialmente en `zed_da3_compare` y tambien hay ideas de `cirtesu_da3_mapping`.

Importante a medio plazo.
No prioritario ahora frente a resultados experimentales.

## Docker y dependencias

Ahora mismo no quiero bloquearme con esto, pero ya hubo avances.

Problema reciente:

bag comprimido zstd -> evaluador necesita Python zstandard -> no estaba instalado

Solucion simple:

grabar sin compresion por ahora. La compresion nacio por querer subir bags a GitHub o moverlos a casa. Para iterar localmente no es esencial.

Permisos ZED:

Antes hacia:

`docker exec --user root ros2-da3-zed-dev chmod 666 /dev/bus/usb/...`

Pero no quiero ir dispositivo por dispositivo. Quiero que el contenedor funcione como un sistema normal, con grupos/permisos correctos.

Cambios hechos en `dockers_cirtesu`:

`docker-compose.yml` usa `group_add` con `HOST_VIDEO_GID` y `HOST_PLUGDEV_GID`.
`.env.example` documenta esos GIDs.
README explica como revisar grupos/permisos.

ZED resources:

El contenedor debe montar:

`${ZED_DATA_VOLUME}/settings:/usr/local/zed/settings`
`${ZED_DATA_VOLUME}/resources:/usr/local/zed/resources`
`${ZED_DATA_VOLUME}/logs:/usr/local/zed/logs`

La optimizacion de modelos neural de ZED puede aparecer como:

`Please wait while the AI model is being optimized for your graphics card`

Eso no es necesariamente una descarga cada vez. Es optimizacion local para la GPU. Deberia persistir si los volumenes estan bien montados y se deja terminar una vez.

Pendiente para mas adelante:

ordenar Dockerfile
ordenar docker compose
asegurar permisos USB de ZED sin chmod manual
asegurar dependencias Python
subir cambios organizados
replicar en portatil

## IMU de la ZED

Pregunta: Si le paso la IMU de la ZED a DA3, podria hacer algo?

DA3, como modelo de depth por imagen/multiview, no usa directamente IMU en el flujo actual.

La IMU podria servir para otras cosas:

- estimar movimiento/orientacion entre frames
- ayudar en visual odometry o SLAM
- validar estabilidad de la camara
- comparar movimiento estimado por ZED contra movimiento estimado por otro metodo 
- sincronizar o etiquetar secuencias con movimiento

Pero no seria simplemente "meter IMU a DA3" y obtener mejor depth, salvo que se construya un pipeline nuevo que use poses externas, secuencias temporales o algun metodo visual-inercial.

Sobre comparar "IMU oficial de la ZED" vs "IMU obtenida por DA3":

- DA3 no obtiene una IMU.
- DA3 puede inferir profundidad y, en algunos modos multivista, puede relacionar vistas/camaras.
- Pero eso no es equivalente a acelerometro/giroscopio.

Lo que si podria compararse mas adelante:

- pose/movimiento de ZED SDK
- pose estimada por visual odometry/SLAM usando imagenes/depth
- poses o relaciones multivista si DA3 expone algo util
- IMU cruda de ZED como referencia de movimiento angular/aceleracion

Para el objetivo actual, IMU no es prioridad.
