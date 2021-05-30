#include <gsl/gsl_math.h>

#include "common.h"
#include "../parameters.h"

// Macro for periodic boundary conditions:
#define periodic(i,N) (((i) >= 0) ? (i) % (N) : (N) - (-(i) % (N)))

/*
 * Inline function for 2D array indexing:
 */
inline int offset2(int i, int j, int N) {
    return periodic(i,N) + N * periodic(j,N);
}

/*
 * Inline function for 3D array indexing,
 * convention is the same as fftw3:
 * http://www.fftw.org/fftw3_doc/Row_002dmajor-Format.html#Row_002dmajor-Format
 */
inline int offset3(int i, int j, int k, int N) {
    return (periodic(i,N) * N + periodic(j,N)) * N + periodic(k,N);
}

/*
 * Stencil coefficients:
 * https://en.wikipedia.org/wiki/Finite_difference_coefficient
 */
dtype laplacian2D(dtype *phi, int i, int j, float dx, int N) {

    dtype laplacian;

    if (globals.StencilOrder == 2) {
        laplacian = (
            (phi[offset2(i+1,j,N)] - 2.0f*phi[offset2(i,j,N)] + phi[offset2(i-1,j,N)])
          + (phi[offset2(i,j+1,N)] - 2.0f*phi[offset2(i,j,N)] + phi[offset2(i,j-1,N)])
          ) / (gsl_pow_2(dx));
    }

    if (globals.StencilOrder == 4) {
        // TODO:
        laplacian = 0.0f;
    }

    return laplacian;
}

dtype laplacian3D(dtype *phi, int i, int j, int k, float dx, int N) {

    dtype laplacian;

    if (globals.StencilOrder == 2) {
        laplacian = (
            (phi[offset3(i+1,j,k,N)] - 2.0f*phi[offset3(i,j,k,N)] + phi[offset3(i-1,j,k,N)])
          + (phi[offset3(i,j+1,k,N)] - 2.0f*phi[offset3(i,j,k,N)] + phi[offset3(i,j-1,k,N)])
          + (phi[offset3(i,j,k+1,N)] - 2.0f*phi[offset3(i,j,k,N)] + phi[offset3(i,j,k-1,N)])
          )/gsl_pow_2(dx);
    }

    if (globals.StencilOrder == 4) {
        // TODO:
        laplacian = 0.0f;
    }

    return laplacian;
}

void gradient2D(dtype *dphi, dtype *phi) {

    int N = globals.N;
    if (globals.StencilOrder == 2) {

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                // TODO: is this the magnitude of the gradient?
                // dphi[i,j] = ((-phi[np.mod(i+2,N),j]+8*phi[np.mod(i+1,N),j]-8*phi[i-1,j] + phi[i-2,j])\
                //     + (-phi[i,np.mod(j+2,N)] + 8*phi[i,np.mod(j+1,N)] -8*phi[i,j-1] + phi[i,j-2]))/(12*dx) 
            }
        }
    }

    if (globals.StencilOrder == 4) {
        // TODO
    }
}