#pragma once

// macros for array indexing:
#define offset2(i,j,N) ((i) + N * (j))
#define offset3(i,j,k,N) (((i) * N + (j)) * N + (k))

// function declarations:

float laplacian2D(float *phi, int i, int j, float dx, int N);
float laplacian3D(float *phi, int i, int j, int k, float dx, int N);

void gradient(float *dphi, float *phi);