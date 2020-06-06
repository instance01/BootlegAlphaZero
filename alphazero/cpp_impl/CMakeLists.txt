cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(GRAB0)

# get and append paths for finding dep
# https://github.com/pytorch/pytorch/issues/17479
#execute_process(
#  COMMAND python -c "import torch; import os; print(os.path.dirname(torch.__file__), end='')"
#  OUTPUT_VARIABLE TorchPath
#)
#list(APPEND CMAKE_PREFIX_PATH ${TorchPath})
list(APPEND CMAKE_PREFIX_PATH "/app/dist/libtorch")

set(Python_ADDITIONAL_VERSIONS 3.7)
find_package(Torch REQUIRED)
find_package(PythonLibs 3 REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
add_definitions("-lstdc++fs")
set(CMAKE_BUILD_TYPE Debug)

add_executable(GRAB0 alphazero.cpp a2c.cpp replay_buffer.cpp env.cpp test_all.cpp)
target_link_libraries(GRAB0 "${TORCH_LIBRARIES}")
target_link_libraries(GRAB0 "${PYTHON_LIBRARIES}")
target_link_libraries(GRAB0 "stdc++fs")
set_property(TARGET GRAB0 PROPERTY CXX_STANDARD 17)
