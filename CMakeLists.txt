project(ncnn-vulkan-compute-sample)
cmake_minimum_required(VERSION 3.10)

set(CMAKE_BUILD_TYPE release)

if(ANDROID)
    set(ncnn_DIR "${CMAKE_CURRENT_SOURCE_DIR}/install-android/lib/cmake/ncnn")
else()
    set(ncnn_DIR "${CMAKE_CURRENT_SOURCE_DIR}/install/lib/cmake/ncnn")
endif()

find_package(ncnn REQUIRED)

add_executable(mla mla.cpp)
target_link_libraries(mla ncnn)
