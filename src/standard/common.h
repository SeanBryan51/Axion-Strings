#pragma once

#ifdef USE_DOUBLE_PRECISION
typedef double dtype;
#else
typedef float dtype;
#endif

// macros for array indexing:
#define offset2(i,j,N) ((i) + N * (j))
#define offset3(i,j,k,N) (((i) * N + (j)) * N + (k))

// evolution.cpp
void kernels(dtype *K1, dtype *K2, dtype *phi1, dtype *phi2, dtype *phidot1, dtype *phidot2);
void apply_drift(dtype *phi1, dtype *phi2, dtype *phi1dot, dtype *phi2dot, dtype *K1, dtype *K2);
void apply_kick(dtype *phi1dot, dtype *phi2dot, dtype *K1, dtype*K2, dtype *K1_next, dtype *K2_next); 

// init.cpp
void init_noise(dtype * phi1, dtype * phi2, dtype * phidot1, dtype *phidot2);
void gaussian_thermal(dtype * phi1, dtype * phi2, dtype * phidot1, dtype *phidot2);

// spatial.cpp
dtype laplacian2D(dtype *phi, int i, int j, float dx, int N);
dtype laplacian3D(dtype *phi, int i, int j, int k, float dx, int N);
void gradient(dtype *dphi, dtype *phi);

// stringID.cpp
