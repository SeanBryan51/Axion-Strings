#pragma once

#ifdef USE_DOUBLE_PRECISION
typedef double dtype;
#else
typedef float dtype;
#endif

// evolution.cpp
void velocity_verlet_scheme(dtype *phi1, dtype *phi2,
                            dtype *phidot1, dtype *phidot2,
                            dtype *ker1_curr, dtype *ker2_curr,
                            dtype *ker1_next, dtype *ker2_next);
void kernels(dtype *K1, dtype *K2, dtype *phi1, dtype *phi2, dtype *phidot1, dtype *phidot2);

// init.cpp
void init_noise(dtype * phi1, dtype * phi2, dtype * phidot1, dtype *phidot2);
void gaussian_thermal(dtype * phi1, dtype * phi2, dtype * phidot1, dtype *phidot2);

// spatial.cpp
inline int offset2(int i, int j, int N);
inline int offset3(int i, int j, int k, int N);
dtype laplacian2D(dtype *phi, int i, int j, float dx, int N);
dtype laplacian3D(dtype *phi, int i, int j, int k, float dx, int N);
void  gradient(dtype *dphi, dtype *phi);

// stringID.cpp
int Cores2D(dtype *field, int thr);
int Cores3D(dtype *field, int thr);