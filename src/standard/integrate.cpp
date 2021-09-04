#include "common.h"

void build_coefficient_matrix(sparse_matrix_t *handle, int NDIMS, int N) {

    // TODO: implement multiple stencil widths

    int length = get_length();

    // COO sparse matrix format data structures, see https://software.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/appendix-a-linear-solvers-basics/sparse-matrix-storage-formats/sparse-blas-coordinate-matrix-storage-format.html for more details.
    // NOTE: the following arrays are deallocated when we call mkl_sparse_destroy()
    MKL_INT nnz;
    MKL_INT *rows;
    MKL_INT *cols;
    dtype *values;

    if (NDIMS == 2) {

        // allocate COO sparse matrix format data structures:
        int num_bands = 1 + 4; // main diagonal + 4 main off diagonals (due to 4 lattice neighbours in 2D)
        nnz = 0;
        rows = (MKL_INT *) calloc(num_bands * length, sizeof(MKL_INT));
        cols = (int *) calloc(num_bands * length, sizeof(MKL_INT));
        values = (dtype *) calloc(num_bands * length, sizeof(dtype));

        assert(rows != NULL && cols != NULL && values != NULL);

        // "fill in" coefficient matrix in COO format:
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < length; i++) {

            int x, y;
            coordinate2(&x, &y, i, N);

            assert(nnz < num_bands * length);
            rows[nnz] = i;
            cols[nnz] = offset2(x,y,N);
            values[nnz] = -4.0f;
            nnz++;

            assert(nnz < num_bands * length);
            rows[nnz] = i;
            cols[nnz] = offset2(x+1,y,N);
            values[nnz] = 1.0f;
            nnz++;

            assert(nnz < num_bands * length);
            rows[nnz] = i;
            cols[nnz] = offset2(x-1,y,N);
            values[nnz] = 1.0f;
            nnz++;

            assert(nnz < num_bands * length);
            rows[nnz] = i;
            cols[nnz] = offset2(x,y+1,N);
            values[nnz] = 1.0f;
            nnz++;

            assert(nnz < num_bands * length);
            rows[nnz] = i;
            cols[nnz] = offset2(x,y-1,N);
            values[nnz] = 1.0f;
            nnz++;
        }
    }

    if (NDIMS == 3) {

        // allocate COO sparse matrix format data structures:
        int num_bands = 1 + 6; // main diagonal + 6 main off diagonals (due to 6 lattice neighbours in 3D)
        nnz = 0;
        rows = (MKL_INT *) calloc(num_bands * length, sizeof(MKL_INT));
        cols = (MKL_INT *) calloc(num_bands * length, sizeof(MKL_INT));
        values = (dtype *) calloc(num_bands * length, sizeof(dtype));

        assert(rows != NULL && cols != NULL && values != NULL);

        // "fill in" coefficient matrix in COO format:
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < length; i++) {

            int x, y, z;
            coordinate3(&x, &y, &z, i, N);

            assert(nnz < num_bands * length);
            rows[nnz] = i;
            cols[nnz] = offset3(x,y,z,N);
            values[nnz] = -6.0f;
            nnz++;

            assert(nnz < num_bands * length);
            rows[nnz] = i;
            cols[nnz] = offset3(x+1,y,z,N);
            values[nnz] = 1.0f;
            nnz++;

            assert(nnz < num_bands * length);
            rows[nnz] = i;
            cols[nnz] = offset3(x-1,y,z,N);
            values[nnz] = 1.0f;
            nnz++;

            assert(nnz < num_bands * length);
            rows[nnz] = i;
            cols[nnz] = offset3(x,y+1,z,N);
            values[nnz] = 1.0f;
            nnz++;

            assert(nnz < num_bands * length);
            rows[nnz] = i;
            cols[nnz] = offset3(x,y-1,z,N);
            values[nnz] = 1.0f;
            nnz++;

            assert(nnz < num_bands * length);
            rows[nnz] = i;
            cols[nnz] = offset3(x,y,z+1,N);
            values[nnz] = 1.0f;
            nnz++;

            assert(nnz < num_bands * length);
            rows[nnz] = i;
            cols[nnz] = offset3(x,y,z-1,N);
            values[nnz] = 1.0f;
            nnz++;
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

void gradient2D(dtype *dphi, dtype *phi) {

    int N = parameters.N;

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            // TODO: is this the magnitude of the gradient?
            // dphi[i,j] = ((-phi[np.mod(i+2,N),j]+8*phi[np.mod(i+1,N),j]-8*phi[i-1,j] + phi[i-2,j])\
            //     + (-phi[i,np.mod(j+2,N)] + 8*phi[i,np.mod(j+1,N)] -8*phi[i,j-1] + phi[i,j-2]))/(12*dx) 
        }
    }
}

/*
 * Velocity-Verlet time evolution algorithm, see equation (125)
 * in arXiv:2006.15122v2 'The art of simulating the early
 * Universe'.
 */
void velocity_verlet_scheme(all_data data) {

    int length = get_length();
    float dt = parameters.time_step;

    // TODO: ker1_curr is zero for the first time step... is this correct?
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < length; i++) {
        data.phi1[i] += dt * (data.phidot1[i] + 0.5f * data.ker1_curr[i] * dt);
        data.phi2[i] += dt * (data.phidot2[i] + 0.5f * data.ker2_curr[i] * dt);
    }

    tau = tau + dt;

    kernels(data.ker1_next, data.ker2_next, data);

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
 * Performs the following element-wise addition to compute the kernel given the fields:
 *  K1 = Laplacian(phi1,dx,N) - 2*(Era/tau)*phidot1 - lambdaPRS*phi1*(phi1**2.0+phi2**2.0 - 1 + (T0/L)**2.0/(3.0*tau**2.0))
 *  K2 = Laplacian(phi2,dx,N) - 2*(Era/tau)*phidot2 - lambdaPRS*phi2*(phi1**2.0+phi2**2.0 - 1 + (T0/L)**2.0/(3.0*tau**2.0))
 */
void kernels(dtype *ker1, dtype *ker2, all_data data) {

    int length = get_length();
    float dx = parameters.space_step;

    mkl_wrapper_sparse_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0f / dx, data.coefficient_matrix, (matrix_descr) { SPARSE_MATRIX_TYPE_SYMMETRIC, SPARSE_FILL_MODE_UPPER, SPARSE_DIAG_NON_UNIT }, data.phi1, 0.0f, ker1);

    mkl_wrapper_sparse_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0f / dx, data.coefficient_matrix, (matrix_descr) { SPARSE_MATRIX_TYPE_SYMMETRIC, SPARSE_FILL_MODE_UPPER, SPARSE_DIAG_NON_UNIT }, data.phi2, 0.0f, ker2);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < length; i++) {
        ker1[i] += - 2.0f / tau * data.phidot1[i]
                   - parameters.lambdaPRS * data.phi1[i] * (
                     pow_2(data.phi1[i]) + pow_2(data.phi2[i]) - 1
                   + pow_2(T_initial) / (3.0f * pow_2(tau / tau_initial))
                   );
        ker2[i] += - 2.0f / tau * data.phidot2[i]
                   - parameters.lambdaPRS * data.phi2[i] * (
                     pow_2(data.phi1[i]) + pow_2(data.phi2[i]) - 1
                   + pow_2(T_initial) / (3.0f * pow_2(tau / tau_initial))
                   );
    }
}