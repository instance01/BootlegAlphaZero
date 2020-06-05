cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(GRAB0)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
set(CMAKE_BUILD_TYPE Debug)

add_executable(GRAB0 a2c.cpp)
target_link_libraries(GRAB0 "${TORCH_LIBRARIES}")
set_property(TARGET GRAB0 PROPERTY CXX_STANDARD 17)
