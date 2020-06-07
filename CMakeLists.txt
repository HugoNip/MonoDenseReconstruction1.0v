cmake_minimum_required(VERSION 3.0)
project(ch12)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11 -march=native -O3")

############### dependencies ######################
# Eigen
Find_Package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})
# OpenCV
find_package(OpenCV 4.0 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
# Sophus
find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})

set(THIRD_PARTY_LIBS ${OpenCV_LIBS} ${Sophus_LIBRARIES})

add_executable(dense_mono dense_mono.cpp)
target_link_libraries(dense_mono ${THIRD_PARTY_LIBS})