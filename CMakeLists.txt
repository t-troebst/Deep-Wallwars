cmake_minimum_required(VERSION 3.26)
project(DeepWallwars CXX)

list(APPEND CMAKE_PREFIX_PATH "libtorch")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Torch REQUIRED)

add_library(core OBJECT
    src/gamestate.cpp
    src/mcts.cpp
    src/simple_policy.cpp
    src/dnn_policy.cpp
    src/play.cpp
)
target_link_libraries(core PUBLIC torch tbb)

add_executable(deep_ww
    src/main.cpp
)
target_link_libraries(deep_ww PRIVATE core)

# Unit tests
find_package(Catch2 3)
if (Catch2_FOUND)
    add_executable(unit_tests
        test/gamestate.cpp
        test/mcts.cpp
    )

    target_link_libraries(unit_tests PRIVATE core Catch2::Catch2WithMain)
    target_include_directories(unit_tests PRIVATE src)
endif()
