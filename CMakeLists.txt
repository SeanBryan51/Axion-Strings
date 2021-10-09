cmake_minimum_required(VERSION 3.12)

message("M E S S A G E  > Hello, world!")

project(
    Axion-Strings
    VERSION 1.0.0
    LANGUAGES CXX
)

# compiler sanity check
message("M E S S A G E  > Compiler ID: ${CMAKE_CXX_COMPILER_ID}")

# specify the C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

enable_testing()
list(APPEND CMAKE_CTEST_ARGUMENTS "--output-on-failure")

# settings determined at compile time:
# note: remember to delete the cmake cache and reconfigure if changing options!

option(AMR_ENABLED "Enable adaptive mesh refinenent." OFF)
if (AMR_ENABLED)
    add_compile_definitions(AMR_ENABLED)
endif()

option(USE_DOUBLE_PRECISION "Enable double precision floating point arithmetic." ON)
if (USE_DOUBLE_PRECISION)
    add_compile_definitions(USE_DOUBLE_PRECISION)
endif()

add_subdirectory(src)