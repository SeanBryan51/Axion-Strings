#include "common.hpp"

sparse_status_t mkl_wrapper_sparse_create_coo (sparse_matrix_t *A, const sparse_index_base_t indexing, const MKL_INT rows, const MKL_INT cols, const MKL_INT nnz, MKL_INT *row_indx, MKL_INT * col_indx, data_t *values) {
#ifdef USE_DOUBLE_PRECISION
    return mkl_sparse_d_create_coo (A, indexing, rows, cols, nnz, row_indx, col_indx, values);
#else
    return mkl_sparse_s_create_coo (A, indexing, rows, cols, nnz, row_indx, col_indx, values);
#endif
}

sparse_status_t mkl_wrapper_sparse_mv (const sparse_operation_t operation, const data_t alpha, const sparse_matrix_t A, const struct matrix_descr descr, const data_t *x, const data_t beta, data_t *y) {
#ifdef USE_DOUBLE_PRECISION
    return mkl_sparse_d_mv (operation, alpha, A, descr, x, beta, y);
#else
    return mkl_sparse_s_mv (operation, alpha, A, descr, x, beta, y);
#endif
}

void mkl_axpy (const MKL_INT n, const data_t a, const data_t *x, const MKL_INT incx, data_t *y, const MKL_INT incy) {
#ifdef USE_DOUBLE_PRECISION
    cblas_daxpy(n, a, x, incx, y, incy);
#else
    cblas_saxpy(n, a, x, incx, y, incy);
#endif
}

void mkl_copy (const MKL_INT n, const data_t *x, const MKL_INT incx, data_t *y, const MKL_INT incy) {
#ifdef USE_DOUBLE_PRECISION
    cblas_dcopy(n, x, incx, y, incy);
#else
    cblas_scopy(n, x, incx, y, incy);
#endif
}

int mkl_v_rng_gaussian(MKL_INT method, VSLStreamStatePtr stream, MKL_INT n, data_t *r, data_t a, data_t sigma) {
#ifdef USE_DOUBLE_PRECISION
    return vdRngGaussian(method, stream, n, r, a, sigma);
#else
    return vsRngGaussian(method, stream, n, r, a, sigma);
#endif
}
