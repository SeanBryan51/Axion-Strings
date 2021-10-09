#include "amr_internal.hpp"

/*
 * Velocity-Verlet (synchronised leapfrog) time evolution algorithm, see equation (125)
 * in arXiv:2006.15122v2 'The art of simulating the early
 * Universe'.
 * Equation of motion is in field rescaled form (alpha = beta = 1):
 *    \phi'' - \nabla^2 \phi + \lambda_{PRS} \tau^{-2} \phi (|\phi|^2 - \tau^2 + T_0^2/3) = 0
 */
void vvsl_field_rescaled() {

    int length = get_length();
    float dt = parameters.time_step;
    float dx = parameters.space_step;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < length; i++) {
        data.phi1[i] += dt * (data.phidot1[i] + 0.5f * data.ker1_curr[i] * dt);
        data.phi2[i] += dt * (data.phidot2[i] + 0.5f * data.ker2_curr[i] * dt);
    }

    tau = tau + dt;

    // compute kernels:

    mkl_wrapper_sparse_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0f / dx, data.coefficient_matrix, (matrix_descr) { SPARSE_MATRIX_TYPE_SYMMETRIC, SPARSE_FILL_MODE_UPPER, SPARSE_DIAG_NON_UNIT }, data.phi1, 0.0f, data.ker1_next);

    mkl_wrapper_sparse_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0f / dx, data.coefficient_matrix, (matrix_descr) { SPARSE_MATRIX_TYPE_SYMMETRIC, SPARSE_FILL_MODE_UPPER, SPARSE_DIAG_NON_UNIT }, data.phi2, 0.0f, data.ker2_next);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < length; i++) {
        data.ker1_next[i] += - 1.0f / pow_2(tau) * parameters.lambdaPRS * data.phi1[i] * (pow_2(data.phi1[i]) + pow_2(data.phi2[i]) - pow_2(tau) + pow_2(T_initial) / (3.0f));
        data.ker2_next[i] += - 1.0f / pow_2(tau) * parameters.lambdaPRS * data.phi2[i] * (pow_2(data.phi1[i]) + pow_2(data.phi2[i]) - pow_2(tau) + pow_2(T_initial) / (3.0f));
    }

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < length; i++) {
        data.phidot1[i] += 0.5f * (data.ker1_curr[i] + data.ker1_next[i]) * dt;
        data.phidot2[i] += 0.5f * (data.ker2_curr[i] + data.ker2_next[i]) * dt;
    }

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < length; i++) {
        data.ker1_curr[i] = data.ker1_next[i];
        data.ker2_curr[i] = data.ker2_next[i];
    }
}

/*
 * Integrates the given level forward in time without knowing the fine level data
 */
void integrate_level(std::vector<level_data> hierarchy, int level) {

    level_data data = hierarchy[level];
    int length = data.size;
    float dx = parameters.space_step;
    float dt = parameters.time_step;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < length; i++) {
        data.phi1[i] += dt * (data.phidot1[i] + 0.5f * data.ker1_curr[i] * dt);
        data.phi2[i] += dt * (data.phidot2[i] + 0.5f * data.ker2_curr[i] * dt);
    }

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < length; i++) {
        data.ker1_next[i] = - 1.0f / pow_2(tau) * parameters.lambdaPRS * data.phi1[i] * (pow_2(data.phi1[i]) + pow_2(data.phi2[i]) - pow_2(tau) + pow_2(T_initial) / (3.0f));
        data.ker2_next[i] = - 1.0f / pow_2(tau) * parameters.lambdaPRS * data.phi2[i] * (pow_2(data.phi1[i]) + pow_2(data.phi2[i]) - pow_2(tau) + pow_2(T_initial) / (3.0f));
    }


}
