cmake_minimum_required(VERSION 2.8.3)
project(space_downsample)
set(CMAKE_CXX_STANDARD 17)

SET(CMAKE_BUILD_TYPE "Debug")

find_package(PCL REQUIRED)

find_package(OpenMP QUIET)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}   ${OpenMP_C_FLAGS}")

include_directories(
  include
  ${PCL_INCLUDE_DIRS}
)
add_executable(space_ds
  src/main.cpp
  include/ikd-Tree/ikd_Tree.cpp
  )
target_link_libraries(space_ds
  ${PCL_LIBRARIES}
  ${OpenMP_CXX_FLAGS}
)
