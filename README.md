# Axion-Strings

## How to compile/run:

Uses cmake to compile. To compile/build the project:

1. Create build directory in project root
```
mkdir build
```
2. Build cmake project
```
cmake -S . -B build
```
2. Run make file to compile executable
```
cd build
make
```
3. Then to run the simulation
```
./build/src/main param.param
```

## How to add new parameters to the parameter file:

1. add parameter to the ```parameters``` struct in parameters.h.
2. add entry to the ```p_list``` array in ```parameters_to_read()``` in utils/read_parameter_file.cpp
3. add parameter to parameter file.
