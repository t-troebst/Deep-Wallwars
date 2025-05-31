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

# ============================================================================
# GUI Support (SFML) - Optional
# ============================================================================

# Add SFML for GUI support
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(SFML QUIET sfml-graphics sfml-window sfml-system)
endif()

# Fallback to find_package if pkg-config didn't work
if(NOT SFML_FOUND)
    find_package(SFML 2.5 COMPONENTS graphics window system QUIET)
endif()

# Set up GUI support
if(SFML_FOUND)
    message(STATUS "SFML found - GUI support enabled")
    set(GUI_ENABLED TRUE)
    
    # Add GUI source files to core library
    target_sources(core PRIVATE
        src/gui/gui_utils.cpp
        src/gui/input_handler.cpp
        src/gui/board_renderer.cpp
        src/gui/game_gui.cpp
    )
    
    # Link SFML libraries - handle both pkg-config and find_package results
    if(SFML_LIBRARIES)
        # pkg-config result
        target_link_libraries(core PUBLIC ${SFML_LIBRARIES})
        if(SFML_INCLUDE_DIRS)
            target_include_directories(core PUBLIC ${SFML_INCLUDE_DIRS})
        endif()
    else()
        # find_package result - use modern targets if available
        if(TARGET sfml-graphics)
            target_link_libraries(core PUBLIC sfml-graphics sfml-window sfml-system)
        else()
            # Fallback to old-style variables
            target_link_libraries(core PUBLIC ${SFML_GRAPHICS_LIBRARY} ${SFML_WINDOW_LIBRARY} ${SFML_SYSTEM_LIBRARY})
            if(SFML_INCLUDE_DIR)
                target_include_directories(core PUBLIC ${SFML_INCLUDE_DIR})
            endif()
        endif()
    endif()
    
    # Enable GUI compilation flag
    target_compile_definitions(core PUBLIC GUI_ENABLED)
else()
    message(STATUS "SFML not found - GUI support disabled. Install libsfml-dev to enable GUI.")
    set(GUI_ENABLED FALSE)
endif()
