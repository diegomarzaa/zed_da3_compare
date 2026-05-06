ros2 launch zed_wrapper zed_camera.launch.py \
  camera_model:=zed \
  param_overrides:="video.publish_left_right:=true"

=========

DA3_MODEL_DIR=/home/usuario/depth_anything_ws/src/Depth-Anything-3/da3_streaming/weights/DA3NESTED-GIANT-LARGE-1.1

ros2 run zed_da3_compare gt_annotation_tool.py --ros-args \
  -p model_dir:="$DA3_MODEL_DIR" \
  -p scene_name:=scene_01_multiple_captures_cirtesuOffice \
  -p left_image_topic:=/zed/zed_node/rgb/color/rect/image \
  -p zed_depth_topic:=/zed/zed_node/depth/depth_registered \
  -p sync_tolerance_ms:=50.0

=========

# Analisis automatico DA3-ZED

python3 /home/usuario/depth_anything_ws/src/zed_da3_compare/zed_da3_compare/analyze_gt_scene.py \
  /home/usuario/depth_anything_ws/src/zed_da3_compare/captures/scene_00_tests

=========

# Notebook Jupyter DA3-ZED

cd /home/usuario/depth_anything_ws/src/zed_da3_compare

jupyter lab notebooks/da3_zed_scene_analysis.ipynb

# Alternativa si no usas JupyterLab:
jupyter notebook notebooks/da3_zed_scene_analysis.ipynb
