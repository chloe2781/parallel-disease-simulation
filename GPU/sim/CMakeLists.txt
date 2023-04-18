cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(gpusim LANGUAGES CXX CUDA)
enable_language(CUDA)

# add include to the header search path
include_directories(include)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CUDA_ARCHITECTURES 61 70 75 80 86)

# add main.cu to the list of sources
file(GLOB SOURCES "src/*.cpp" "src/*.cu" "src/main.cu")

# for curand
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# create an executable from the sources
add_executable(gpusim ${SOURCES})
target_link_libraries(gpusim ${CUDA_LIBRARIES})
