#include "common.h"

/*
 * Velocity-Verlet time evolution algorithm, see equation (125)
 * in arXiv:2006.15122v2 'The art of simulating the early
 * Universe'.
 */
void velocity_verlet_scheme(all_data data) {

    int N = parameters.N;
    int length = (parameters.NDIMS == 3) ? (N * N * N) : (N * N);
    float dt = parameters.time_step;

    // TODO: ker1_curr is zero for the first time step... is this correct?
    for (int i = 0; i < length; i++) {
        data.phi1[i] += dt * (data.phidot1[i] + 0.5f * data.ker1_curr[i] * dt);
        data.phi2[i] += dt * (data.phidot2[i] + 0.5f * data.ker2_curr[i] * dt);
    }

    tau = tau + dt;

    kernels(data.ker1_next, data.ker2_next, data);

    for (int i = 0; i < length; i++) {
        data.phidot1[i] += 0.5f * (data.ker1_curr[i] + data.ker1_next[i]) * dt;
        data.phidot2[i] += 0.5f * (data.ker2_curr[i] + data.ker2_next[i]) * dt;
    }

    for (int i = 0; i < length; i++) {
        data.ker1_curr[i] = data.ker1_next[i];
        data.ker2_curr[i] = data.ker2_next[i];
    }
}

/*
 * Performs the following element-wise addition to compute the kernel given the fields:
 *  K1 = Laplacian(phi1,dx,N) - 2*(Era/tau)*phidot1 - lambdaPRS*phi1*(phi1**2.0+phi2**2.0 - 1 + (T0/L)**2.0/(3.0*tau**2.0))
 *  K2 = Laplacian(phi2,dx,N) - 2*(Era/tau)*phidot2 - lambdaPRS*phi2*(phi1**2.0+phi2**2.0 - 1 + (T0/L)**2.0/(3.0*tau**2.0))
 */
void kernels(dtype *ker1, dtype *ker2, all_data data) {

    int N = parameters.N;
    int length = (parameters.NDIMS == 3) ? (N * N * N) : (N * N);
    float dx = parameters.space_step;

    mkl_wrapper_sparse_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0f / dx, data.coefficient_matrix, (matrix_descr) { SPARSE_MATRIX_TYPE_SYMMETRIC, SPARSE_FILL_MODE_UPPER, SPARSE_DIAG_NON_UNIT }, data.phi1, 0.0f, ker1);

    mkl_wrapper_sparse_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0f / dx, data.coefficient_matrix, (matrix_descr) { SPARSE_MATRIX_TYPE_SYMMETRIC, SPARSE_FILL_MODE_UPPER, SPARSE_DIAG_NON_UNIT }, data.phi2, 0.0f, ker2);

    for (int i = 0; i < length; i++) {
        ker1[i] = ker1[i] - 2.0f / tau * data.phidot1[i]
                          - parameters.lambdaPRS * data.phi1[i] * (
                              gsl_pow_2(data.phi1[i]) + gsl_pow_2(data.phi2[i]) - 1
                            + gsl_pow_2(T_initial) / (3.0f * gsl_pow_2(tau / tau_initial))
                          );
        ker2[i] = ker2[i] - 2.0f / tau * data.phidot2[i]
                          - parameters.lambdaPRS * data.phi2[i] * (
                              gsl_pow_2(data.phi1[i]) + gsl_pow_2(data.phi2[i]) - 1
                            + gsl_pow_2(T_initial) / (3.0f * gsl_pow_2(tau / tau_initial))
                          );
    }
}