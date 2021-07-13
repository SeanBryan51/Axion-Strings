#include "common.h"
#include "../parameters.h"

#include "mkl.h"

/*
 * Velocity-Verlet time evolution algorithm, see equation (125)
 * in arXiv:2006.15122v2 'The art of simulating the early
 * Universe'.
 */
void velocity_verlet_scheme(dtype *phi1, dtype *phi2,
                            dtype *phidot1, dtype *phidot2,
                            dtype *ker1_curr, dtype *ker2_curr,
                            dtype *ker1_next, dtype *ker2_next) {

    int N = parameters.N;
    int length = (parameters.NDIMS == 3) ? (N * N * N) : (N * N);
    float dt = parameters.time_step;

    for (int i = 0; i < length; i++) {
        phi1[i] += dt * (phidot1[i] + 0.5f * ker1_curr[i] * dt);
        phi2[i] += dt * (phidot2[i] + 0.5f * ker2_curr[i] * dt);
    }

    t_evol = t_evol + dt;

    kernels(ker1_next, ker2_next, phi1, phi2, phidot1, phidot2);

    for (int i = 0; i < length; i++) {
        phidot1[i] += 0.5f * (ker1_curr[i] + ker1_next[i]) * dt;
        phidot2[i] += 0.5f * (ker2_curr[i] + ker2_next[i]) * dt;
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

    int N = parameters.N;
    int length = (parameters.NDIMS == 3) ? (N * N * N) : (N * N);
    float dx = parameters.space_step;

    mkl_wrapper_sparse_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0f / dx, coefficient_matrix, (matrix_descr) { SPARSE_MATRIX_TYPE_SYMMETRIC, SPARSE_FILL_MODE_UPPER, SPARSE_DIAG_NON_UNIT }, phi1, 0.0f, ker1);

    mkl_wrapper_sparse_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0f / dx, coefficient_matrix, (matrix_descr) { SPARSE_MATRIX_TYPE_SYMMETRIC, SPARSE_FILL_MODE_UPPER, SPARSE_DIAG_NON_UNIT }, phi2, 0.0f, ker2);

    for (int i = 0; i < length; i++) {
        ker1[i] = ker1[i] - 2.0f / t_evol * phidot1[i]
                          - parameters.lambdaPRS * phi1[i] * (
                              gsl_pow_2(phi1[i]) + gsl_pow_2(phi2[i]) - 1
                            + gsl_pow_2(T_initial) / (3.0f * gsl_pow_2(t_evol / t_initial))
                          );
        ker2[i] = ker2[i] - 2.0f / t_evol * phidot2[i]
                          - parameters.lambdaPRS * phi2[i] * (
                              gsl_pow_2(phi1[i]) + gsl_pow_2(phi2[i]) - 1
                            + gsl_pow_2(T_initial) / (3.0f * gsl_pow_2(t_evol / t_initial))
                          );
    }
}