add_executable(
    main
    main.cpp
    parameters.h
)

# link third party libraries:

include_directories(/usr/local/Cellar/fftw/3.3.8_2/include)
link_directories(/usr/local/Cellar/fftw/3.3.8_2/lib)
target_link_options(main PUBLIC -I/usr/local/Cellar/fftw/3.3.8_2/include -L/usr/local/Cellar/fftw/3.3.8_2/lib -lfftw3 -lm)

find_package(GSL REQUIRED)
target_link_libraries(main GSL::gsl)

# define custom libraries:

add_library(
    standard
    standard/common.h
    standard/interface.h
    standard/run.cpp
    standard/init.cpp
    standard/spatial.cpp
    standard/evolve.cpp
    # standard/stringID.cpp
)

target_include_directories(standard PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(standard GSL::gsl)
target_link_libraries(main standard)

add_library(
    utils
    utils/utils.h
    utils/fileutil.cpp
    utils/read_parameters.cpp
)

target_include_directories(utils PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(utils GSL::gsl)
target_link_libraries(main utils)

# configure tests:

add_executable(test_standard tests/test_standard.cpp)
target_link_libraries(test_standard standard utils)
add_test(test-standard test_standard)