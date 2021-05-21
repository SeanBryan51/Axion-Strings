cmake_minimum_required(VERSION 3.1.3)

message("M E S S A G E  > Hello, world!")

project(
    Axion-Strings-AMR
    VERSION 1.0.0
    LANGUAGES CXX
)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_subdirectory(src)