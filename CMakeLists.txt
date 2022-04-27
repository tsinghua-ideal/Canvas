# Project settings
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(Canvas)

# C++ standards
set(CMAKE_CXX_STANDARD 17)

# Fetch external dependencies
include(FetchContent)

# Fetch pybind11
FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11
        GIT_TAG        v2.9.2
)

# Includes and libraries
include_directories(3rd-party)
include_directories(include)

# Add subdirectory of pybind11
FetchContent_GetProperties(pybind11)
if(NOT pybind11_POPULATED)
    FetchContent_Populate(pybind11)
    add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR})
endif()

# Sub-directories
add_subdirectory(src)
