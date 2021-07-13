#include <stdio.h>
#include <stdlib.h>
#include <gsl/gsl_math.h>
#include <assert.h>

#include "standard/common.h"

int is_equal(dtype x, dtype y, dtype tolerance) {
    assert(tolerance > 0.0f);
    return x - y <= tolerance && x - y >= -tolerance;
}

void status_check(sparse_status_t status) {
    switch (status) {
    case SPARSE_STATUS_SUCCESS:
        printf("\nThe operation was successful.\n");
        break;
    case SPARSE_STATUS_NOT_INITIALIZED:
        printf("\nThe routine encountered an empty handle or matrix array.\n");
        break;
    case SPARSE_STATUS_ALLOC_FAILED:
        printf("\nInternal memory allocation failed.\n");
        break;
    case SPARSE_STATUS_INVALID_VALUE:
        printf("\nThe input parameters contain an invalid value.\n");
        break;
    case SPARSE_STATUS_EXECUTION_FAILED:
        printf("\nExecution failed.\n");
        break;
    case SPARSE_STATUS_INTERNAL_ERROR:
        printf("\nAn error in algorithm implementation occurred.\n");
        break;
    case SPARSE_STATUS_NOT_SUPPORTED:
        printf("\nThe requested operation is not supported.\n");
    default:
        break;
    }
}

void test_build_coefficient_matrix() {

    // success case: 2D N*N by N*N coefficient matrix with N=4
    int N = 4;
    sparse_matrix_t mat_coo, mat_csr;
    sparse_status_t status;
    dtype correct_matrix[] = {
        -4, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
        1, -4, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
        0, 1, -4, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0,
        1, 0, 1, -4, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1,
        1, 0, 0, 0, -4, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 1, -4, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 1, -4, 1, 0, 0, 1, 0, 0, 0, 0, 0,
        0, 0, 0, 1, 1, 0, 1, -4, 0, 0, 0, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 0, 0, -4, 1, 0, 1, 1, 0, 0, 0,
        0, 0, 0, 0, 0, 1, 0, 0, 1, -4, 1, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 0, 0, 1, 0, 0, 1, -4, 1, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, -4, 0, 0, 0, 1,
        1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, -4, 1, 0, 1,
        0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, -4, 1, 0,
        0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, -4, 1,
        0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, -4
    };

    build_coefficient_matrix(&mat_coo, 2, N);

    status = mkl_sparse_convert_csr (mat_coo, SPARSE_OPERATION_NON_TRANSPOSE, &mat_csr);
    // status_check(status);
    assert(status == SPARSE_STATUS_SUCCESS); // check conversion was successful

    int rows, cols;
    sparse_index_base_t indexing = SPARSE_INDEX_BASE_ZERO;
    int *rows_start;
    int *rows_end;
    int *col_indx;

#ifdef USE_DOUBLE_PRECISION
    double *values;
    mkl_sparse_d_export_csr (mat_csr, &indexing, &rows, &cols, &rows_start, &rows_end, &col_indx, &values);
#else
    float *values;
    mkl_sparse_s_export_csr (mat_csr, &indexing, &rows, &cols, &rows_start, &rows_end, &col_indx, &values);
#endif

    assert(rows == cols && rows == N*N);

    int n_values_read = 0;
    int n_nonzero = rows_end[rows-1];
    for (int i = 0; i < N*N; i++) {
        for (int j = 0; j < N*N; j++) {
            float val = 0.0f;
            if (col_indx[n_values_read] == j) {
                val = values[n_values_read];
                n_values_read++;
            }
            assert(val == correct_matrix[offset2(i,j,N*N)]);
            // printf("%.1f ", val);
        }
        // printf("\n");
    }

    status = mkl_sparse_destroy(mat_coo);
    // status_check(status);
    assert(status == SPARSE_STATUS_SUCCESS); // check successfully destroyed
}

void test_fdm_matrix2D() {

    int N;
    dtype *phi;
    dtype *laplacian;
    sparse_matrix_t mat;
    sparse_status_t status;

    // success case: simple array of zeros
    N = 128;
    phi = (dtype *) calloc(N * N, sizeof(dtype));
    laplacian = (dtype *) calloc(N * N, sizeof(dtype));
    build_coefficient_matrix(&mat, 2, N);

    status = mkl_wrapper_sparse_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0f, mat, (matrix_descr) { SPARSE_MATRIX_TYPE_SYMMETRIC, SPARSE_FILL_MODE_UPPER, SPARSE_DIAG_NON_UNIT }, phi, 0.0f, laplacian);
    status_check(status);
    assert(status == SPARSE_STATUS_SUCCESS);

    for (int i = 0; i < N*N; i++) {
        assert(is_equal(laplacian[i], 0.0f, 1e-30f));
    }

    mkl_sparse_destroy(mat);
    status_check(status);
    assert(status == SPARSE_STATUS_SUCCESS); // check successfully destroyed

    free(phi);
    free(laplacian);

    // success case: wave with periodic boundary conditions
    N = 256;
    phi = (dtype *) calloc(N * N, sizeof(dtype));
    laplacian = (dtype *) calloc(N * N, sizeof(dtype));
    build_coefficient_matrix(&mat, 2, N);

    dtype L = N / 2.0f;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            // define field values
            phi[offset2(i,j,N)] = sinf(M_PI / L * i) + cosf(M_PI / L * j);
        }
    }

    status = mkl_wrapper_sparse_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0f, mat, (matrix_descr) { SPARSE_MATRIX_TYPE_SYMMETRIC, SPARSE_FILL_MODE_UPPER, SPARSE_DIAG_NON_UNIT }, phi, 0.0f, laplacian);
    status_check(status);
    assert(status == SPARSE_STATUS_SUCCESS);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            // compare discretised laplacian and actual laplacian
            dtype discretised_laplacian = laplacian[offset2(i,j,N)];
            dtype actual_laplacian = - gsl_pow_2(M_PI / L) * (sinf(M_PI / L * i) + cosf(M_PI / L * j));
            assert(is_equal(discretised_laplacian, actual_laplacian, 1e-5f));
        }
    }

    status = mkl_sparse_destroy(mat);
    status_check(status);
    assert(status == SPARSE_STATUS_SUCCESS);

    free(phi);
    free(laplacian);
}

void test_fdm_matrix3D() {

    int N;
    dtype *phi;
    dtype *laplacian;
    sparse_matrix_t mat;
    sparse_status_t status;

    // success case: simple array of zeros
    N = 128;
    phi = (dtype *) calloc(N * N * N, sizeof(dtype));
    laplacian = (dtype *) calloc(N * N * N, sizeof(dtype));
    build_coefficient_matrix(&mat, 3, N);

    status = mkl_wrapper_sparse_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0f, mat, (matrix_descr) { SPARSE_MATRIX_TYPE_SYMMETRIC, SPARSE_FILL_MODE_UPPER, SPARSE_DIAG_NON_UNIT }, phi, 0.0f, laplacian);
    status_check(status);
    assert(status == SPARSE_STATUS_SUCCESS);

    for (int i = 0; i < N*N*N; i++) {
        assert(is_equal(laplacian[i], 0.0f, 1e-30f));
    }

    mkl_sparse_destroy(mat);
    status_check(status);
    assert(status == SPARSE_STATUS_SUCCESS); // check successfully destroyed

    free(phi);
    free(laplacian);

    // success case: wave with periodic boundary conditions
    N = 256;
    phi = (dtype *) calloc(N * N * N, sizeof(dtype));
    laplacian = (dtype *) calloc(N * N * N, sizeof(dtype));
    build_coefficient_matrix(&mat, 3, N);

    dtype L = N / 2.0f;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                // define field values
                phi[offset3(i,j,k,N)] = sinf(M_PI / L * i) + cosf(M_PI / L * j) + sinf(M_PI / L * k);
            }
        }
    }

    status = mkl_wrapper_sparse_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0f, mat, (matrix_descr) { SPARSE_MATRIX_TYPE_SYMMETRIC, SPARSE_FILL_MODE_UPPER, SPARSE_DIAG_NON_UNIT }, phi, 0.0f, laplacian);
    status_check(status);
    assert(status == SPARSE_STATUS_SUCCESS);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                // compare discretised laplacian and actual laplacian
                dtype discretised_laplacian = laplacian[offset3(i,j,k,N)];
                dtype actual_laplacian = - gsl_pow_2(M_PI / L) * (sinf(M_PI / L * i) + cosf(M_PI / L * j) + sinf(M_PI / L * k));
                assert(is_equal(discretised_laplacian, actual_laplacian, 1e-5f));
            }
        }
    }

    status = mkl_sparse_destroy(mat);
    status_check(status);
    assert(status == SPARSE_STATUS_SUCCESS);

    free(phi);
    free(laplacian);

}

int main(void) {

    test_build_coefficient_matrix();
    test_fdm_matrix2D();
    test_fdm_matrix3D();

    return EXIT_SUCCESS;
}