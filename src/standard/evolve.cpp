#include <stdio.h>
#include <stdlib.h>
#include <gsl/gsl_math.h>

#include "../parameters.h"
#include "spatial.h"

/*
 * Performs the element wise addition:
 * phi1 = phi1 + dtau*(phidot1 + 0.5*K1*dtau)
 * phi2 = phi2 + dtau*(phidot2 + 0.5*K2*dtau)
 */
void apply_drift(float *phi1, float *phi2, float *phi1dot, float *phi2dot, float *K1, float *K2) {

    int N = globals.N;
    if (globals.NDIMS == 2) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                phi1[offset2(i,j,N)] += globals.dtau * (phi1dot[offset2(i,j,N)] + 0.5f * K1[offset2(i,j,N)] * globals.dtau);
                phi2[offset2(i,j,N)] += globals.dtau * (phi2dot[offset2(i,j,N)] + 0.5f * K2[offset2(i,j,N)] * globals.dtau);
            }
        }
    }
    if (globals.NDIMS == 3) {
        // TODO:

    }
}

/*
 * Performs the element wise addition:
 *  phidot1 = phidot1 + 0.5*(K1 + K1_next)*dtau
 *  phidot2 = phidot2 + 0.5*(K2 + K2_next)*dtau
 */
void apply_kick(float *phi1dot, float *phi2dot, float *K1, float*K2, float *K1_next, float *K2_next) {
    int N = globals.N;
    if (globals.NDIMS == 2) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                phi1dot[offset2(i,j,N)] += 0.5f * (K1[offset2(i,j,N)] + K1_next[offset2(i,j,N)]) * globals.dtau;
                phi2dot[offset2(i,j,N)] += 0.5f * (K2[offset2(i,j,N)] + K2_next[offset2(i,j,N)]) * globals.dtau;
            }
        }
    }

    if (globals.NDIMS == 3) {
        // TODO:

    }

}

/*
 * Performs the following element-wise addition to compute the kernel given the fields:
 *  K1 = Laplacian(phi1,dx,N) - 2*(Era/t_evol)*phidot1 - lambdaPRS*phi1*(phi1**2.0+phi2**2.0 - 1 + (T0/L)**2.0/(3.0*t_evol**2.0))
 *  K2 = Laplacian(phi2,dx,N) - 2*(Era/t_evol)*phidot2 - lambdaPRS*phi2*(phi1**2.0+phi2**2.0 - 1 + (T0/L)**2.0/(3.0*t_evol**2.0))
 */
void kernels(float *K1, float *K2, float *phi1, float *phi2, float *phidot1, float *phidot2) {

    int N = globals.N;

    if (globals.NDIMS == 2) {

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                // TODO: write a separate function for the potential:
                K1[offset2(i,j,N)] = laplacian2D(phi1,i,j) -2.0f * globals.Era / globals.t_evol * phidot1[offset2(i,j,N)] - globals.lambdaPRS * phi1[offset2(i,j,N)] * (gsl_pow_2(phi1[offset2(i,j,N)]) + gsl_pow_2(phi2[offset2(i,j,N)]) - 1 + gsl_pow_2(globals.T0/globals.L) / (3.0f * gsl_pow_2(globals.t_evol)));

                K2[offset2(i,j,N)] = laplacian2D(phi2,i,j) -2.0f * globals.Era / globals.t_evol * phidot2[offset2(i,j,N)] - globals.lambdaPRS * phi2[offset2(i,j,N)] * (gsl_pow_2(phi1[offset2(i,j,N)]) + gsl_pow_2(phi2[offset2(i,j,N)]) - 1 + gsl_pow_2(globals.T0/globals.L) / (3.0f * gsl_pow_2(globals.t_evol)));
            }
        }
    }

    if (globals.NDIMS == 3) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                for (int k = 0; k < N; k++) {
                    K1[offset3(i,j,k,N)] = laplacian3D(phi1,i,j,k) - 2.0f * (globals.Era / globals.t_evol) * phidot1[offset3(i,j,k,N)] - globals.lambdaPRS * phi1[offset3(i,j,k,N)] * (gsl_pow_2(phi1[offset3(i,j,k,N)]) + gsl_pow_2(phi2[offset3(i,j,k,N)]) - 1 + gsl_pow_2(globals.T0/globals.L) / (3.0f * gsl_pow_2(globals.t_evol)));
                    K2[offset3(i,j,k,N)] = laplacian3D(phi2,i,j,k) - 2.0f * (globals.Era / globals.t_evol) * phidot2[offset3(i,j,k,N)] - globals.lambdaPRS * phi2[offset3(i,j,k,N)] * (gsl_pow_2(phi1[offset3(i,j,k,N)]) + gsl_pow_2(phi2[offset3(i,j,k,N)]) - 1 + gsl_pow_2(globals.T0/globals.L) / (3.0f * gsl_pow_2(globals.t_evol)));
                }
            }
        }
    }
}