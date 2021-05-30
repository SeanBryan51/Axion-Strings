#include <gsl/gsl_math.h>

#include "common.h"
#include "../parameters.h"

// TODO: implement as a macro to avoid function overhead?
int periodic(int i, int N) {
    if (i < 0) {
        return N - (-i % N);
    } else if (i == 0) {
        return 0;
    } else {
        return i % N;
    }
}

/*
 * Stencil coefficients:
 * https://en.wikipedia.org/wiki/Finite_difference_coefficient
 */
dtype laplacian2D(dtype *phi, int i, int j, float dx, int N) {

    dtype laplacian;

    if (globals.StencilOrder == 2) {
        laplacian = (
            (phi[offset2(periodic(i+1,N),j,N)] - 2.0f*phi[offset2(i,j,N)] + phi[offset2(periodic(i-1,N),j,N)])
          + (phi[offset2(i,periodic(j+1,N),N)] - 2.0f*phi[offset2(i,j,N)] + phi[offset2(i,periodic(j-1,N),N)])
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
            (phi[offset3(periodic(i+1,N),j,k,N)] - 2.0f*phi[offset3(i,j,k,N)] + phi[offset3(periodic(i-1,N),j,k,N)])
          + (phi[offset3(i,periodic(j+1,N),k,N)] - 2.0f*phi[offset3(i,j,k,N)] + phi[offset3(i,periodic(j-1,N),k,N)])
          + (phi[offset3(i,j,periodic(k+1,N),N)] - 2.0f*phi[offset3(i,j,k,N)] + phi[offset3(i,j,periodic(k-1,N),N)])
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