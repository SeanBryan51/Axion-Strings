add_executable(
    main
    main.cpp
    common.hpp

    physics.cpp
    init.cpp
    string_finding.cpp
    mkl_wrapper.cpp

    utils/fileio.cpp
    utils/read_parameters.cpp
)

# link third party libraries:

include_directories(/usr/local/Cellar/fftw/3.3.8_2/include)
link_directories(/usr/local/Cellar/fftw/3.3.8_2/lib)
target_link_options(main PUBLIC -I/usr/local/Cellar/fftw/3.3.8_2/include -L/usr/local/Cellar/fftw/3.3.8_2/lib -lfftw3_omp -lfftw3 -lm)

set(BLA_VENDOR Intel10_64lp)
find_package(BLAS REQUIRED)
target_link_libraries(main ${BLAS_LIBRARIES})

find_package(OpenMP REQUIRED) 
target_link_libraries(main OpenMP::OpenMP_CXX)

# define custom libraries:

add_library(
    standard
    standard/s_interface.hpp
    standard/s_internal.hpp
    standard/s_run.cpp
    standard/s_integrate.cpp
    standard/s_util.cpp
)

target_include_directories(standard PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(standard OpenMP::OpenMP_CXX)
target_link_libraries(standard ${BLAS_LIBRARIES})
target_link_libraries(main standard)


# configure tests:

# add_executable(laplacian tests/laplacian.cpp)
# target_link_libraries(laplacian standard utils)
# add_test(test-laplacian laplacian)

# add_executable(fdm_matrix tests/fdm_matrix.cpp)
# target_link_libraries(fdm_matrix standard utils)
# add_test(test-fdm_matrix fdm_matrix)
