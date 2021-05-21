# Axion-Strings

## How to compile/run:

Uses cmake to compile. To compile/build the project run

```
cmake .
```

```
cmake --build .
```

Then to run the simulation
```
./build/src/main param.param
```

## How to add new parameters to the parameter file:

1. add parameter to the ```globals``` struct in parameters.h.
2. add entry to the ```parameters``` array in ```parameters_to_read()``` in utils/read_parameter_file.cpp
3. add parameter to parameter file.
