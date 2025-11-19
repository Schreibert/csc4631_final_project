cmake_minimum_required(VERSION 3.14)

# Core library
add_library(balatro_core STATIC
    src/card.cpp
    src/hand_eval.cpp
    src/scoring.cpp
    src/blind_state.cpp
    src/simulator.cpp
)

target_include_directories(balatro_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Compiler warnings
if(MSVC)
    target_compile_options(balatro_core PRIVATE /W4)
else()
    target_compile_options(balatro_core PRIVATE -Wall -Wextra -Wpedantic)
endif()

# Tests
if(BUILD_TESTS)
    enable_testing()

    # Fetch Google Test
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )
    # For Windows: Prevent overriding the parent project's compiler/linker settings
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)

    add_subdirectory(tests)
endif()
