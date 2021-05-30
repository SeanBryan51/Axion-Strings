#include <stdio.h>
#include <stdlib.h>
#include <gsl/gsl_math.h>

#include "../parameters.h"
#include "common.h"

/*
 * Velocity-Verlet time evolution algorithm, see equation (125)
 * in arXiv:2006.15122v2 'The art of simulating the early
 * Universe'.
 */
void velocity_verlet_scheme(dtype *phi1, dtype *phi2,
                            dtype *phidot1, dtype *phidot2,
                            dtype *ker1_curr, dtype *ker2_curr,
                            dtype *ker1_next, dtype *ker2_next) {

    int N = globals.N;
    int length = (globals.NDIMS == 3) ? (N * N * N) : (N * N);

    for (int i = 0; i < length; i++) {
        phi1[i] += globals.dtau * (phidot1[i] + 0.5f * ker1_curr[i] * globals.dtau);
        phi2[i] += globals.dtau * (phidot2[i] + 0.5f * ker2_curr[i] * globals.dtau);
    }

    globals.t_evol = globals.t_evol + globals.dtau;

    kernels(ker1_next, ker2_next, phi1, phi2, phidot1, phidot2);

    for (int i = 0; i < length; i++) {
        phidot1[i] += 0.5f * (ker1_curr[i] + ker1_next[i]) * globals.dtau;
        phidot2[i] += 0.5f * (ker2_curr[i] + ker2_next[i]) * globals.dtau;
    }

    for (int i = 0; i < length; i++) {
        ker1_curr[i] = ker1_next[i];
        ker2_curr[i] = ker2_next[i];
    }
}

/*
 * Performs the following element-wise addition to compute the kernel given the fields:
 *  K1 = Laplacian(phi1,dx,N) - 2*(Era/t_evol)*phidot1 - lambdaPRS*phi1*(phi1**2.0+phi2**2.0 - 1 + (T0/L)**2.0/(3.0*t_evol**2.0))
 *  K2 = Laplacian(phi2,dx,N) - 2*(Era/t_evol)*phidot2 - lambdaPRS*phi2*(phi1**2.0+phi2**2.0 - 1 + (T0/L)**2.0/(3.0*t_evol**2.0))
 */
void kernels(dtype *ker1, dtype *ker2, dtype *phi1, dtype *phi2, dtype *phidot1, dtype *phidot2) {

    int N = globals.N;
    float dx = globals.dx;
    dtype l_phi1, l_phi2; // (temporary variables)

    if (globals.NDIMS == 2) {

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {

                l_phi1 = laplacian2D(phi1, i, j, dx, N);
                ker1[offset2(i,j,N)] = (
                    l_phi1 - 2.0f * globals.Era / globals.t_evol * phidot1[offset2(i,j,N)]
                  - globals.lambdaPRS * phi1[offset2(i,j,N)] * (
                        gsl_pow_2(phi1[offset2(i,j,N)]) + gsl_pow_2(phi2[offset2(i,j,N)]) - 1
                      + gsl_pow_2(globals.T0/globals.L) / (3.0f * gsl_pow_2(globals.t_evol))
                  ));

                l_phi2 = laplacian2D(phi2, i, j, dx, N);
                ker2[offset2(i,j,N)] = (
                    l_phi2 - 2.0f * globals.Era / globals.t_evol * phidot2[offset2(i,j,N)]
                  - globals.lambdaPRS * phi2[offset2(i,j,N)] * (
                        gsl_pow_2(phi1[offset2(i,j,N)]) + gsl_pow_2(phi2[offset2(i,j,N)]) - 1
                      + gsl_pow_2(globals.T0/globals.L) / (3.0f * gsl_pow_2(globals.t_evol))
                  ));
            }
        }
    }

    if (globals.NDIMS == 3) {

        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                for (int k = 0; k < N; k++) {
                    l_phi1 = laplacian3D(phi1, i, j, k, dx, N);
                    ker1[offset3(i,j,k,N)] = (
                        l_phi1 - 2.0f * (globals.Era / globals.t_evol) * phidot1[offset3(i,j,k,N)]
                      - globals.lambdaPRS * phi1[offset3(i,j,k,N)] * (
                            gsl_pow_2(phi1[offset3(i,j,k,N)]) + gsl_pow_2(phi2[offset3(i,j,k,N)]) - 1
                          + gsl_pow_2(globals.T0/globals.L) / (3.0f * gsl_pow_2(globals.t_evol))
                      ));

                    l_phi2 = laplacian3D(phi2, i, j, k, dx, N);
                    ker2[offset3(i,j,k,N)] = (
                        l_phi2 - 2.0f * (globals.Era / globals.t_evol) * phidot2[offset3(i,j,k,N)]
                      - globals.lambdaPRS * phi2[offset3(i,j,k,N)] * (
                            gsl_pow_2(phi1[offset3(i,j,k,N)]) + gsl_pow_2(phi2[offset3(i,j,k,N)]) - 1
                          + gsl_pow_2(globals.T0/globals.L) / (3.0f * gsl_pow_2(globals.t_evol))
                      ));
                }
            }
        }
    }
}