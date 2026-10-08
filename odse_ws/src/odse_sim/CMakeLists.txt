cmake_minimum_required(VERSION 3.16)
project(odse_sim NONE)

find_package(ament_cmake REQUIRED)

install(DIRECTORY config launch urdf worlds DESTINATION share/${PROJECT_NAME})

ament_package()
