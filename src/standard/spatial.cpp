#include "common.h"
#include "../parameters.h"
#include "../utils/utils.h"

#include <assert.h>

sparse_matrix_t coefficient_matrix;

void set_coefficient_matrix(char *file_path, sparse_matrix_t *handle) {

    int length, nnz;
    // NOTE: the following arrays are deallocated when we call mkl_sparse_destroy()
    int *rows;
    int *cols;
    dtype *values;

    read_mtx_file(file_path, &length, &nnz, rows, cols, values);

    // create sparse matrix object in COO format:
    sparse_status_t status = mkl_wrapper_sparse_create_coo (handle, SPARSE_INDEX_BASE_ZERO, length, length, nnz, rows, cols, values);
    assert(status == SPARSE_STATUS_SUCCESS);
}

void build_coefficient_matrix(sparse_matrix_t *handle, int NDIMS, int N) {

    // TODO: implement multiple stencil widths

    assert(NDIMS == 2 || NDIMS == 3);

    // COO sparse matrix format data structures, see https://software.intel.com/content/www/us/en/develop/documentation/onemkl-developer-reference-c/top/appendix-a-linear-solvers-basics/sparse-matrix-storage-formats/sparse-blas-coordinate-matrix-storage-format.html for more details.
    // NOTE: the following arrays are deallocated when we call mkl_sparse_destroy()
    int length; // length of solution vector
    MKL_INT nnz;
    MKL_INT *rows;
    MKL_INT *cols;
    dtype *values;

    if (NDIMS == 2) {
        length = N * N;

        // allocate COO sparse matrix format data structures:
        int num_bands = 1 + 4; // main diagonal + 4 main off diagonals (due to 4 lattice neighbours in 2D)
        nnz = 0;
        rows = (MKL_INT *) calloc(num_bands * length, sizeof(MKL_INT));
        cols = (int *) calloc(num_bands * length, sizeof(MKL_INT));
        values = (dtype *) calloc(num_bands * length, sizeof(dtype));

        // "fill in" coefficient matrix in COO format:
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
        length = N * N * N;

        // allocate COO sparse matrix format data structures:
        int num_bands = 1 + 6; // main diagonal + 6 main off diagonals (due to 6 lattice neighbours in 3D)
        nnz = 0;
        rows = (MKL_INT *) calloc(num_bands * length, sizeof(MKL_INT));
        cols = (MKL_INT *) calloc(num_bands * length, sizeof(MKL_INT));
        values = (dtype *) calloc(num_bands * length, sizeof(dtype));

        // "fill in" coefficient matrix in COO format:
        for (int i = 0; i < length; i++) {

            int x, y, z;
            coordinate3(&x, &y, &z, i, N);

            rows[nnz] = i;
            cols[nnz] = offset3(x,y,z,N);
            values[nnz] = -6.0f;
            nnz++;

            rows[nnz] = i;
            cols[nnz] = offset3(x+1,y,z,N);
            values[nnz] = 1.0f;
            nnz++;

            rows[nnz] = i;
            cols[nnz] = offset3(x-1,y,z,N);
            values[nnz] = 1.0f;
            nnz++;

            rows[nnz] = i;
            cols[nnz] = offset3(x,y+1,z,N);
            values[nnz] = 1.0f;
            nnz++;

            rows[nnz] = i;
            cols[nnz] = offset3(x,y-1,z,N);
            values[nnz] = 1.0f;
            nnz++;

            rows[nnz] = i;
            cols[nnz] = offset3(x,y,z+1,N);
            values[nnz] = 1.0f;
            nnz++;

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
      ) / (gsl_pow_2(dx));

    return laplacian;
}

dtype laplacian3D(dtype *phi, int i, int j, int k, float dx, int N) {

    dtype laplacian;

    laplacian = (
        (phi[offset3(i+1,j,k,N)] - 2.0f*phi[offset3(i,j,k,N)] + phi[offset3(i-1,j,k,N)])
      + (phi[offset3(i,j+1,k,N)] - 2.0f*phi[offset3(i,j,k,N)] + phi[offset3(i,j-1,k,N)])
      + (phi[offset3(i,j,k+1,N)] - 2.0f*phi[offset3(i,j,k,N)] + phi[offset3(i,j,k-1,N)])
      )/gsl_pow_2(dx);

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
