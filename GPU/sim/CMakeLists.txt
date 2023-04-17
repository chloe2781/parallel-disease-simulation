cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(gpusim LANGUAGES CXX CUDA)
enable_language(CUDA)

# add include to the header search path
include_directories(include)
file(GLOB SOURCES "src/*.cpp" "src/*.cu")
set(CMAKE_CUDA_COMPILER "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1/bin/nvcc.exe")
add_executable(gpusim ${SOURCES})
