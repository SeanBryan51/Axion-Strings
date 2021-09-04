#include "common.h"

#include "mkl.h"

sparse_status_t mkl_wrapper_sparse_create_coo (sparse_matrix_t *A, const sparse_index_base_t indexing, const MKL_INT rows, const MKL_INT cols, const MKL_INT nnz, MKL_INT *row_indx, MKL_INT * col_indx, dtype *values) {
#ifdef USE_DOUBLE_PRECISION
    return mkl_sparse_d_create_coo (A, indexing, rows, cols, nnz, row_indx, col_indx, values);
#else
    return mkl_sparse_s_create_coo (A, indexing, rows, cols, nnz, row_indx, col_indx, values);
#endif
}

sparse_status_t mkl_wrapper_sparse_mv (const sparse_operation_t operation, const dtype alpha, const sparse_matrix_t A, const struct matrix_descr descr, const dtype *x, const dtype beta, dtype *y) {
#ifdef USE_DOUBLE_PRECISION
    return mkl_sparse_d_mv (operation, alpha, A, descr, x, beta, y);
#else
    return mkl_sparse_s_mv (operation, alpha, A, descr, x, beta, y);
#endif
}

void mkl_axpy (const MKL_INT n, const dtype a, const dtype *x, const MKL_INT incx, dtype *y, const MKL_INT incy) {
#ifdef USE_DOUBLE_PRECISION
    cblas_daxpy(n, a, x, incx, y, incy);
#else
    cblas_saxpy(n, a, x, incx, y, incy);
#endif
}

void mkl_copy (const MKL_INT n, const dtype *x, const MKL_INT incx, dtype *y, const MKL_INT incy) {
#ifdef USE_DOUBLE_PRECISION
    cblas_dcopy(n, x, incx, y, incy);
#else
    cblas_scopy(n, x, incx, y, incy);
#endif
}

#if 0
void mkl_v_add(int n, const dtype *a, const dtype *b, dtype *r) {
#ifdef USE_DOUBLE_PRECISION
    vdAdd(n, a, b, r);
#else
    vsAdd(n, a, b, r);
#endif
}
#endif