#include <gsl/gsl_math.h>

#include "spatial.h"
#include "../parameters.h"

// TODO: implement as a macro to avoid function overhead?
static int periodic(int i, int N) {
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
float laplacian2D(float *phi, int i, int j) {

    int N = globals.N;
    float laplacian;

    if (globals.StencilOrder == 2) {
        laplacian = (( phi[offset2(periodic(i+1,N),j,N)] - 2.0f*phi[offset2(i,j,N)] + phi[offset2(periodic(i-1,N),j,N)]) + (( phi[offset2(i,periodic(j+1,N),N)] - 2.0f*phi[offset2(i,j,N)] + phi[offset2(i,periodic(j-1,N),N)] )))/(gsl_pow_2(globals.dx));
    }

    if (globals.StencilOrder == 4) {
        // TODO:
        laplacian = 0.0f;
    }

    if (globals.StencilOrder == 6) {
        // TODO:
        laplacian = 0.0f;
    }

    return laplacian;
}

float laplacian3D(float *phi, int i, int j, int k) {

    int N = globals.N;
    float laplacian;

    if (globals.StencilOrder == 2) {
        // TODO:
        laplacian = 0.0f;
    }

    if (globals.StencilOrder == 4) {
        // TODO:
        laplacian = 0.0f;
    }

    if (globals.StencilOrder == 6) {
        // TODO:
        laplacian = 0.0f;
    }

    return laplacian;
}

void gradient(float *dphi, float *phi) {

    if (globals.NDIMS == 2) {
        if (globals.StencilOrder == 2) {

            int N = globals.N;
            for (int i = 0; i < N; i++) {
                for (int j = 0; j < N; j++) {
                    // TODO:
                    // dphi[i,j] = ((-phi[np.mod(i+2,N),j]+8*phi[np.mod(i+1,N),j]-8*phi[i-1,j] + phi[i-2,j])\
                    //     + (-phi[i,np.mod(j+2,N)] + 8*phi[i,np.mod(j+1,N)] -8*phi[i,j-1] + phi[i,j-2]))/(12*dx) 
                }
            }
        }

        if (globals.StencilOrder == 4) {
            // TODO
        }

        if (globals.StencilOrder == 6) {
            // TODO
        }

    }

    if (globals.NDIMS == 3) {
        // TODO:
    }
}