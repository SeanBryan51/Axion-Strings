add_executable(
    main
    main.cpp
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
    common
    common/common.hpp
    common/physics.cpp
    common/init.cpp
    common/string_finding.cpp
    common/mkl_wrapper.cpp
    common/fileio.cpp
    common/read_parameters.cpp
)

target_include_directories(common PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(common OpenMP::OpenMP_CXX)
target_link_libraries(common ${BLAS_LIBRARIES})
target_link_libraries(main common)

add_library(
    amr
    amr/amr_interface.hpp
    amr/amr_internal.hpp
    amr/amr_run.cpp
    amr/amr_integrate.cpp
    amr/amr_point_clustering.cpp
)

target_include_directories(amr PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(amr OpenMP::OpenMP_CXX)
target_link_libraries(amr ${BLAS_LIBRARIES})
target_link_libraries(amr common)
target_link_libraries(main amr)

add_library(
    standard
    standard/s_interface.hpp
    standard/s_internal.hpp
    standard/s_run.cpp
    standard/s_integrate.cpp
)

target_include_directories(standard PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(standard OpenMP::OpenMP_CXX)
target_link_libraries(standard ${BLAS_LIBRARIES})
target_link_libraries(standard common)
target_link_libraries(main standard)

# configure tests:

add_executable(laplacian tests/laplacian.cpp)
target_link_libraries(laplacian common)
target_link_libraries(laplacian standard)
add_test(test-laplacian laplacian)

add_executable(fdm_matrix tests/fdm_matrix.cpp)
target_link_libraries(fdm_matrix common)
target_link_libraries(fdm_matrix standard)
add_test(test-fdm_matrix fdm_matrix)
