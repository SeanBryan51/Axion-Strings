#include "common.h"

void build_coefficient_matrix(sparse_matrix_t *handle, int NDIMS, int N) {

    int stencil_setting = parameters.stencil_setting;
    std::vector<std::vector<dtype>> coefficients = {
        {        -2.0f * NDIMS,      1.0f },                                       // stencil_setting = 0
        {   -5.0f/2.0f * NDIMS, 4.0f/3.0f, -1.0f/12.0f },                          // stencil_setting = 1
        { -49.0f/18.0f * NDIMS, 3.0f/2.0f, -3.0f/20.0f,  1.0f/90.0f },             // stencil_setting = 2
        {-205.0f/72.0f * NDIMS, 8.0f/5.0f,  -1.0f/5.0f, 8.0f/315.0f, -1.0f/560.0f} // stencil_setting = 3
    };

    int length = (NDIMS == 3) ? (N * N * N) : (N * N);

    // COO sparse matrix format data structures, see https://software.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/appendix-a-linear-solvers-basics/sparse-matrix-storage-formats/sparse-blas-coordinate-matrix-storage-format.html for more details.
    // NOTE: the following arrays are deallocated when we call mkl_sparse_destroy()
    MKL_INT nnz;
    MKL_INT *rows;
    MKL_INT *cols;
    dtype *values;

    if (NDIMS == 2) {

        // allocate COO sparse matrix format data structures:
        int num_bands = 1 + 4 * (coefficients[stencil_setting].size() - 1); // main diagonal + 4 * stencil (due to 4 lattice neighbours in 2D)
        nnz = 0;
        rows = (MKL_INT *) calloc(num_bands * length, sizeof(MKL_INT));
        cols = (int *) calloc(num_bands * length, sizeof(MKL_INT));
        values = (dtype *) calloc(num_bands * length, sizeof(dtype));

        assert(rows != NULL && cols != NULL && values != NULL);

        // "fill in" coefficient matrix in COO format:
        // Note: don't try to parallelise this loop!
        for (int i = 0; i < length; i++) {

            int x, y;
            coordinate2(&x, &y, i, N);

            // central node:
            assert(nnz < num_bands * length);
            rows[nnz] = i;
            cols[nnz] = offset2(x,y,N);
            values[nnz] = coefficients[stencil_setting][0];
            nnz++;

            // neighbouring nodes:
            for (int l = 1; l < coefficients[stencil_setting].size(); l++) {
                assert(nnz < num_bands * length);
                rows[nnz] = i;
                cols[nnz] = offset2(x+l,y,N);
                values[nnz] = coefficients[stencil_setting][l];
                nnz++;

                assert(nnz < num_bands * length);
                rows[nnz] = i;
                cols[nnz] = offset2(x-l,y,N);
                values[nnz] = coefficients[stencil_setting][l];
                nnz++;

                assert(nnz < num_bands * length);
                rows[nnz] = i;
                cols[nnz] = offset2(x,y+l,N);
                values[nnz] = coefficients[stencil_setting][l];
                nnz++;

                assert(nnz < num_bands * length);
                rows[nnz] = i;
                cols[nnz] = offset2(x,y-l,N);
                values[nnz] = coefficients[stencil_setting][l];
                nnz++;
            }

        }
    }

    if (NDIMS == 3) {

        // allocate COO sparse matrix format data structures:
        int num_bands = 1 + 6 * (coefficients[stencil_setting].size() - 1); // main diagonal + 6 * stencil (due to 6 lattice neighbours in 3D)
        nnz = 0;
        rows = (MKL_INT *) calloc(num_bands * length, sizeof(MKL_INT));
        cols = (MKL_INT *) calloc(num_bands * length, sizeof(MKL_INT));
        values = (dtype *) calloc(num_bands * length, sizeof(dtype));

        assert(rows != NULL && cols != NULL && values != NULL);

        // "fill in" coefficient matrix in COO format:
        // Note: don't try to parallelise this loop!
        for (int i = 0; i < length; i++) {

            int x, y, z;
            coordinate3(&x, &y, &z, i, N);

            // central node:
            assert(nnz < num_bands * length);
            rows[nnz] = i;
            cols[nnz] = offset3(x,y,z,N);
            values[nnz] = coefficients[stencil_setting][0];
            nnz++;

            // neighbouring nodes:
            for (int l = 1; l < coefficients[stencil_setting].size(); l++) {
                assert(nnz < num_bands * length);
                rows[nnz] = i;
                cols[nnz] = offset3(x+l,y,z,N);
                values[nnz] = coefficients[stencil_setting][l];
                nnz++;

                assert(nnz < num_bands * length);
                rows[nnz] = i;
                cols[nnz] = offset3(x-l,y,z,N);
                values[nnz] = coefficients[stencil_setting][l];
                nnz++;

                assert(nnz < num_bands * length);
                rows[nnz] = i;
                cols[nnz] = offset3(x,y+l,z,N);
                values[nnz] = coefficients[stencil_setting][l];
                nnz++;

                assert(nnz < num_bands * length);
                rows[nnz] = i;
                cols[nnz] = offset3(x,y-l,z,N);
                values[nnz] = coefficients[stencil_setting][l];
                nnz++;

                assert(nnz < num_bands * length);
                rows[nnz] = i;
                cols[nnz] = offset3(x,y,z+l,N);
                values[nnz] = coefficients[stencil_setting][l];
                nnz++;

                assert(nnz < num_bands * length);
                rows[nnz] = i;
                cols[nnz] = offset3(x,y,z-l,N);
                values[nnz] = coefficients[stencil_setting][l];
                nnz++;
            }
        }
    }

    // create sparse matrix object:
    sparse_status_t status = mkl_wrapper_sparse_create_coo (handle, SPARSE_INDEX_BASE_ZERO, length, length, nnz, rows, cols, values);
    assert(status == SPARSE_STATUS_SUCCESS);
}

/*
 * Stencil coefficients:
 * https://en.wikipedia.org/wiki/Finite_difference_coefficient
 */
dtype laplacian2D(dtype *phi, int i, int j, float dx, int N) {

    dtype laplacian;

    laplacian = (
        (phi[offset2(i+1,j,N)] - 2.0f*phi[offset2(i,j,N)] + phi[offset2(i-1,j,N)])
      + (phi[offset2(i,j+1,N)] - 2.0f*phi[offset2(i,j,N)] + phi[offset2(i,j-1,N)])
      ) / (pow_2(dx));

    return laplacian;
}

dtype laplacian3D(dtype *phi, int i, int j, int k, float dx, int N) {

    dtype laplacian;

    laplacian = (
        (phi[offset3(i+1,j,k,N)] - 2.0f*phi[offset3(i,j,k,N)] + phi[offset3(i-1,j,k,N)])
      + (phi[offset3(i,j+1,k,N)] - 2.0f*phi[offset3(i,j,k,N)] + phi[offset3(i,j-1,k,N)])
      + (phi[offset3(i,j,k+1,N)] - 2.0f*phi[offset3(i,j,k,N)] + phi[offset3(i,j,k-1,N)])
      )/pow_2(dx);

    return laplacian;
}

/*
 * Compute the kernel given the fields. The kernel follows from the field
 * rescaled equation of motion:
 *    K[\phi] = \nabla^2 \phi - \lambda_{PRS} \tau^{-2} \phi (|\phi|^2 - \tau^2 + T_0^2/3)
 */
static void kernels_field_rescaled(dtype *ker1, dtype *ker2, all_data data) {

    int length = get_length();
    float dx = parameters.space_step;

    mkl_wrapper_sparse_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0f / dx, data.coefficient_matrix, (matrix_descr) { SPARSE_MATRIX_TYPE_SYMMETRIC, SPARSE_FILL_MODE_UPPER, SPARSE_DIAG_NON_UNIT }, data.phi1, 0.0f, ker1);

    mkl_wrapper_sparse_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0f / dx, data.coefficient_matrix, (matrix_descr) { SPARSE_MATRIX_TYPE_SYMMETRIC, SPARSE_FILL_MODE_UPPER, SPARSE_DIAG_NON_UNIT }, data.phi2, 0.0f, ker2);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < length; i++) {
        ker1[i] += - 1.0f / pow_2(tau) * parameters.lambdaPRS * data.phi1[i] * (
                     pow_2(data.phi1[i]) + pow_2(data.phi2[i]) - pow_2(tau)
                   + pow_2(T_initial) / (3.0f)
                   );
        ker2[i] += - 1.0f / pow_2(tau) * parameters.lambdaPRS * data.phi2[i] * (
                     pow_2(data.phi1[i]) + pow_2(data.phi2[i]) - pow_2(tau)
                   + pow_2(T_initial) / (3.0f)
                   );
    }
}

/*
 * Velocity-Verlet (synchronised leapfrog) time evolution algorithm, see equation (125)
 * in arXiv:2006.15122v2 'The art of simulating the early
 * Universe'.
 * Equation of motion is in field rescaled form (alpha = beta = 1):
 *    \phi'' - \nabla^2 \phi + \lambda_{PRS} \tau^{-2} \phi (|\phi|^2 - \tau^2 + T_0^2/3) = 0
 */
void vvsl_field_rescaled(all_data data) {

    int length = get_length();
    float dt = parameters.time_step;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < length; i++) {
        data.phi1[i] += dt * (data.phidot1[i] + 0.5f * data.ker1_curr[i] * dt);
        data.phi2[i] += dt * (data.phidot2[i] + 0.5f * data.ker2_curr[i] * dt);
    }

    tau = tau + dt;

    kernels_field_rescaled(data.ker1_next, data.ker2_next, data);

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
 * Compute the kernel given the fields. The kernel follows from the Hamiltonian-like
 * equation of motion (see Eq. (219-221) in arXiv:2006.15122v2):
 *    K[\phi] = \tau^2 \nabla^2 \phi - \lambda_{PRS} \tau^{2} \phi (|\phi|^2 - 1 + T_0^2/(3 \tau^2))
 * Note: in this case, phidot is really pi where pi = phidot * a^2.
 */
static void kernels_hamiltonian_form(dtype *ker1, dtype *ker2, all_data data) {

    int length = get_length();
    float dx = parameters.space_step;

    mkl_wrapper_sparse_mv(SPARSE_OPERATION_NON_TRANSPOSE, pow_2(tau) / dx, data.coefficient_matrix, (matrix_descr) { SPARSE_MATRIX_TYPE_SYMMETRIC, SPARSE_FILL_MODE_UPPER, SPARSE_DIAG_NON_UNIT }, data.phi1, 0.0f, ker1);

    mkl_wrapper_sparse_mv(SPARSE_OPERATION_NON_TRANSPOSE, pow_2(tau) / dx, data.coefficient_matrix, (matrix_descr) { SPARSE_MATRIX_TYPE_SYMMETRIC, SPARSE_FILL_MODE_UPPER, SPARSE_DIAG_NON_UNIT }, data.phi2, 0.0f, ker2);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < length; i++) {
        ker1[i] += - pow_2(tau) * parameters.lambdaPRS * data.phi1[i] * (
                     pow_2(data.phi1[i]) + pow_2(data.phi2[i]) - 1.0f
                   + pow_2(T_initial) / (3.0f * pow_2(tau))
                   );
        ker2[i] += - pow_2(tau) * parameters.lambdaPRS * data.phi2[i] * (
                     pow_2(data.phi1[i]) + pow_2(data.phi2[i]) - 1.0f
                   + pow_2(T_initial) / (3.0f * pow_2(tau))
                   );
    }
}

/*
 * Velocity-Verlet (synchronised leapfrog) time evolution algorithm, see equation (125)
 * in arXiv:2006.15122v2 'The art of simulating the early
 * Universe'.
 * Equation of motion is in the Hamiltonian like form (see Eq. (219-221) in arXiv:2006.15122v2)
 *    \pi' - \tau^2 \nabla^2 \phi + \lambda_{PRS} \tau^{2} \phi (|\phi|^2 - 1 + T_0^2/(3 \tau^2)) = 0
 * Note: in this case, phidot is really pi where pi = phidot * a^2.
 */
void vvsl_hamiltonian_form(all_data data) {

    int length = get_length();
    float dt = parameters.time_step;

    tau = tau + dt;

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < length; i++) {
        data.phi1[i] += dt / pow_2(tau) * (data.phidot1[i] + 0.5f * dt * data.ker1_curr[i]);
        data.phi2[i] += dt / pow_2(tau) * (data.phidot2[i] + 0.5f * dt * data.ker2_curr[i]);
    }

    kernels_hamiltonian_form(data.ker1_next, data.ker2_next, data);

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
