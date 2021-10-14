# Axion-Strings

## How to compile/run:

Uses cmake to compile. To compile/build the project:

1. Create build directory in project root
```
mkdir build
```
2. Generate build system
```
cmake -S . -B build
```
2. Build / compile
```cmake --build build
```
3. Then to run the simulation
```
./build/src/main default.param
```

## How to add new parameters to the parameter file:

1. add parameter to the ```parameters``` struct in common/common.h.
2. add entry to the ```p_list``` array in ```parameters_to_read()``` in common/read_parameter_file.cpp
3. add parameter to parameter file.
