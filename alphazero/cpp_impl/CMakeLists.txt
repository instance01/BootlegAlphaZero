cmake_minimum_required(VERSION 3.6 FATAL_ERROR)
include(ExternalProject)
project(GRAB0)

find_program(MAKE_EXE NAMES gmake nmake make)
ExternalProject_Add(tensorboard_logger
  PREFIX            dist/
  GIT_REPOSITORY    https://github.com/instance01/tensorboard_logger.git
  CONFIGURE_COMMAND ""
  BUILD_IN_SOURCE   1
  INSTALL_COMMAND   ""
  UPDATE_COMMAND    ""
  BUILD_COMMAND     ${MAKE_EXE} all
)


list(APPEND CMAKE_PREFIX_PATH "/app/dist/libtorch")

set(Python_ADDITIONAL_VERSIONS 3.7)
find_package(Protobuf REQUIRED)
find_package(PythonLibs 3 REQUIRED)
find_package(Torch REQUIRED)

include_directories(${PYTHON_INCLUDE_DIRS} "dist/src/tensorboard_logger/include")
link_directories("dist/src/tensorboard_logger/")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
add_definitions("-lstdc++fs")
set(CMAKE_BUILD_TYPE Debug)

add_executable(${PROJECT_NAME} alphazero.cpp a2c.cpp replay_buffer.cpp env.cpp test_all.cpp tensorboard_util.cpp simple_thread_pool.cpp mcts.cpp)
target_link_libraries(${PROJECT_NAME} "${TORCH_LIBRARIES}")
target_link_libraries(${PROJECT_NAME} "${PYTHON_LIBRARIES}")
target_link_libraries(${PROJECT_NAME} "stdc++fs")
target_link_libraries(${PROJECT_NAME} "-ltensorboard_logger")
target_link_libraries(${PROJECT_NAME} "-lprotobuf -pthread")
set_property(TARGET ${PROJECT_NAME} PROPERTY CXX_STANDARD 17)
add_dependencies(${PROJECT_NAME} tensorboard_logger)
