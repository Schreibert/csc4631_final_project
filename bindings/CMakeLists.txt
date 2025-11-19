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

target_link_libraries(_balatro_core PRIVATE balatro_core)

# Install to python package directory
install(TARGETS _balatro_core DESTINATION ${CMAKE_SOURCE_DIR}/python/balatro_env)
