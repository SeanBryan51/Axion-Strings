#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <gsl/gsl_math.h>

#include "mkl_spblas.h"

#ifdef USE_DOUBLE_PRECISION
typedef double dtype;
#else
typedef float dtype;
#endif

// Macro for periodic boundary conditions:
#define periodic(i,N) (((i) >= 0) ? (i) % (N) : (N) - (-(i) % (N)))

/*
 * Inline function for 2D array indexing:
 */
inline int offset2(int i, int j, int N) {
    return periodic(i,N) + N * periodic(j,N);
}

/*
 * Inline function that performs the inverse operation of offset2().
 * Returns the (i,j) coordinate corresponding to a given offset.
 */
inline void coordinate2(int *i, int *j, int offset, int N) {
    *j = offset / N;
    *i = offset % N;
}

/*
 * Inline function for 3D array indexing,
 * convention is the same as fftw3:
 * http://www.fftw.org/fftw3_doc/Row_002dmajor-Format.html#Row_002dmajor-Format
 */
inline int offset3(int i, int j, int k, int N) {
    return (periodic(i,N) * N + periodic(j,N)) * N + periodic(k,N);
}

/*
 * Inline function that performs the inverse operation of offset3().
 * Returns the (i,j,k) coordinate corresponding to a given offset.
 */
inline void coordinate3(int *i, int *j, int *k, int offset, int N) {
    *i = offset / (N * N);
    int layer = offset - (*i) * N * N;
    *j = layer / N;
    *k = layer % N;
}

// physics.cpp
extern float t_evol;
extern float t_initial;
extern float T_initial;
extern float t_phys_initial;
extern float R_initial;
extern float reduced_planck_mass;
extern float m_saxion;
extern float g_star;
extern float m_eff_squared;
extern float light_crossing_time;
void set_internal_variables();
float physical_time(float t_conformal);
float scale_factor(float t_conformal);
float hubble_parameter(float t_conformal);
float temperature(float t_conformal);
float string_tension(float t_confomal);
float meff_squared(float t_conformal);

// evolution.cpp
void velocity_verlet_scheme(dtype *phi1, dtype *phi2,
                            dtype *phidot1, dtype *phidot2,
                            dtype *ker1_curr, dtype *ker2_curr,
                            dtype *ker1_next, dtype *ker2_next);
void kernels(dtype *K1, dtype *K2, dtype *phi1, dtype *phi2, dtype *phidot1, dtype *phidot2);

// init.cpp
void init_noise(dtype * phi1, dtype * phi2, dtype * phidot1, dtype *phidot2);
void gaussian_thermal(dtype * phi1, dtype * phi2, dtype * phidot1, dtype *phidot2);

// spatial.cpp
extern sparse_matrix_t coefficient_matrix;
void  set_coefficient_matrix(char *file_path, sparse_matrix_t *handle);
void  build_coefficient_matrix(sparse_matrix_t *handle, int NDIMS, int N);
dtype laplacian2D(dtype *phi, int i, int j, float dx, int N);
dtype laplacian3D(dtype *phi, int i, int j, int k, float dx, int N);
void  gradient(dtype *dphi, dtype *phi);

// stringID.cpp
int Cores2D(dtype *field, int thr);
int Cores3D(dtype *field, int thr);

// mkl_wrapper.cpp
sparse_status_t mkl_wrapper_sparse_create_coo (sparse_matrix_t *A, const sparse_index_base_t indexing, const MKL_INT rows, const MKL_INT cols, const MKL_INT nnz, MKL_INT *row_indx, MKL_INT * col_indx, dtype *values);
sparse_status_t mkl_wrapper_sparse_mv (const sparse_operation_t operation, const dtype alpha, const sparse_matrix_t A, const struct matrix_descr descr, const dtype *x, const dtype beta, dtype *y);
