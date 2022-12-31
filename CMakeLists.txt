# Project settings.
cmake_minimum_required(VERSION 3.11 FATAL_ERROR)
project(Canvas)

# C++ standards.
set(CMAKE_CXX_STANDARD 17)

# Compiler flags.
add_definitions(-fPIC)

# Customized release build type with assertions: RelBuildWithAsserts
set(CMAKE_C_FLAGS_RELBUILDWITHASSERTS "-O2 -g")
set(CMAKE_CXX_FLAGS_RELBUILDWITHASSERTS "-O2 -g")

# Fetch external dependencies.
include(FetchContent)

# Fetch pybind11.
# TODO: may refer to the official repo.
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://gitee.com/lkhlll/pybind11
    GIT_TAG        v2.9.2
)

# Boost.
find_package(Boost REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

# GTest.
enable_testing()
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

# Add subdirectory of pybind11.
FetchContent_GetProperties(pybind11)
if(NOT pybind11_POPULATED)
    FetchContent_Populate(pybind11)
    add_subdirectory(${pybind11_SOURCE_DIR} ${pybind11_BINARY_DIR})
endif()

# Includes and libraries.
include_directories(3rd-party)
include_directories(include)

# Sub-directories.
add_subdirectory(src)
add_subdirectory(tests)
