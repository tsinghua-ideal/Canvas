cmake_minimum_required(VERSION 3.4)
project(Canvas)

set(CMAKE_CXX_STANDARD 17)

include(FetchContent)


# Fetch pybind11
FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11
        GIT_TAG        v2.9.2
)

# Add subdirectory of pybind11
FetchContent_GetProperties(pybind11)
if(NOT pybind11_POPULATED)
    FetchContent_Populate(pybind11)
    add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR})
endif()


# Link C++ API entry
pybind11_add_module(cc_canvas ccsrc/cc_api.cpp)
