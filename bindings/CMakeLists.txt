cmake_minimum_required(VERSION 3.14)

# Find Python and pybind11
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.11.1
)
FetchContent_MakeAvailable(pybind11)

# Create Python module
pybind11_add_module(_balatro_core pybind/bindings.cpp)

# Ensure the pybind11 targets (and their include dirs) are linked to the extension
target_link_libraries(_balatro_core PRIVATE pybind11::module balatro_core)

# Auto-copy .pyd to python package after build (for development)
add_custom_command(TARGET _balatro_core POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:_balatro_core>
        ${CMAKE_SOURCE_DIR}/python/balatro_env/$<TARGET_FILE_NAME:_balatro_core>
    COMMENT "Copying Python module to python/balatro_env/"
)

# Install to python package directory (for installation)
install(TARGETS _balatro_core DESTINATION ${CMAKE_SOURCE_DIR}/python/balatro_env)
