cmake_minimum_required(VERSION 3.6 FATAL_ERROR)
project(GRAB0)

list(APPEND CMAKE_PREFIX_PATH "/app/dist/libtorch")

set(Python_ADDITIONAL_VERSIONS 3.7)
find_package(Protobuf REQUIRED)
find_package(PythonLibs 3 REQUIRED)
find_package(Torch REQUIRED)

include_directories(${PYTHON_INCLUDE_DIRS} "dist/tensorboard_logger/include")
link_directories("dist/tensorboard_logger")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
add_definitions("-lstdc++fs")
set(CMAKE_BUILD_TYPE Debug)

add_executable(${PROJECT_NAME} alphazero.cpp a2c.cpp replay_buffer.cpp env.cpp test_all.cpp tensorboard_util.cpp)
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")
target_link_libraries(${PROJECT_NAME} "${PYTHON_LIBRARIES}")
target_link_libraries(${PROJECT_NAME} "stdc++fs")
target_link_libraries(${PROJECT_NAME} "-ltensorboard_logger")
target_link_libraries(${PROJECT_NAME} "-lprotobuf -pthread")
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)
