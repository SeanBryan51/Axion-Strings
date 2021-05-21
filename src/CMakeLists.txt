add_executable(
    main
    main.cpp
    parameters.h
)

# link third party libraries:

find_package(GSL REQUIRED)
target_link_libraries(main GSL::gsl)

# define custom libraries:

add_library(
    standard
    standard/run.cpp
    standard/interface.h
    standard/init.cpp
    standard/init.h
    standard/spatial.cpp
    standard/spatial.h
    standard/evolve.cpp
    standard/evolve.h
    # standard/stringID.cpp
    # standard/stringID.h
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
