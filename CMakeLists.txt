cmake_minimum_required(VERSION 3.8)
project(zed_da3_compare)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(sensor_msgs REQUIRED)

ament_python_install_package(${PROJECT_NAME})

install(PROGRAMS
  zed_da3_compare/da3_depth_node.py
  zed_da3_compare/da3_depth_multicam_node.py
  zed_da3_compare/da3_offline_bag_eval.py
  zed_da3_compare/da3_stereo_depth_node.py
  zed_da3_compare/gt_annotation_tool.py
  zed_da3_compare/interactive_depth_viewer.py
  zed_da3_compare/zed_sync_sampler_node.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
