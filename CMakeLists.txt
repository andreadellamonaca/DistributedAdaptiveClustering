cmake_minimum_required(VERSION 3.15)
project(DistributedAdaptiveClustering)

set(CMAKE_CXX_STANDARD 14)

find_package(Armadillo REQUIRED)
include_directories(${ARMADILLO_INCLUDE_DIR})

add_executable(DistributedAdaptiveClustering main.cpp adaptive_clustering.h adaptive_clustering.cpp)

target_link_libraries(DistributedAdaptiveClustering m igraph ${ARMADILLO_LIBRARIES})