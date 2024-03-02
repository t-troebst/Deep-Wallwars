cmake_minimum_required(VERSION 3.26)
project(DeepWallwars CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)

find_package(CUDAToolkit REQUIRED)
find_package(gflags REQUIRED)
find_package(glog REQUIRED)
find_package(folly REQUIRED)
# TODO: find TensorRT with CMake

add_library(core OBJECT
    src/batched_model.cpp
    src/batched_model_policy.cpp
    src/cached_policy.cpp
    src/cuda_wrappers.cpp
    src/gamestate.cpp
    src/game_recorder.cpp
    src/mcts.cpp
    src/model.cpp
    src/play.cpp
    src/simple_policy.cpp
    src/state_conversions.cpp
    src/tensorrt_model.cpp
)
target_link_libraries(core PUBLIC Folly::folly CUDA::cudart nvinfer nvonnxparser atomic)

add_executable(deep_ww
    src/main.cpp
)
target_link_libraries(deep_ww PRIVATE core)

# Unit tests (optional)
find_package(Catch2 3)
if (Catch2_FOUND)
    add_executable(unit_tests
        test/batched_model.cpp
        test/gamestate.cpp
        test/main.cpp
        test/mcts.cpp
    )

    target_link_libraries(unit_tests PRIVATE core Catch2::Catch2)
    target_include_directories(unit_tests PRIVATE src)
endif()
