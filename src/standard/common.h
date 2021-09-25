#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <vector>

#include <omp.h>

#include "mkl.h"
#include "mkl_vsl.h"
#include "mkl_spblas.h"

#include "../parameters.h"

#ifdef USE_DOUBLE_PRECISION
typedef double dtype;
#else
typedef float dtype;
#endif

// Macro for periodic boundary conditions:
#define periodic(i,N) (((i) >= 0) ? (i) % (N) : (N) - (-(i) % (N)))

// Mega struct containing all the pointers to large arrays/solution vectors:
typedef struct _all_data {
    dtype *phi1; // phi_1 field values
    dtype *phi2; // phi_2 field values
    dtype *phidot1; // phi_1 time derivative
    dtype *phidot2; // phi_2 time derivative
    dtype *ker1_curr; // current kernel for phi_1 equation of motion
    dtype *ker2_curr; // current kernel for phi_2 equation of motion
    dtype *ker1_next; // next kernel for phi_1 equation of motion
    dtype *ker2_next; // next kernel for phi_2 equation of motion

    dtype *axion; // axion field values
    dtype *saxion; // saxion field values

    sparse_matrix_t coefficient_matrix;

} all_data;

typedef struct vec2i { int x; int y; } vec2i;
typedef struct vec3i { int x; int y; int z; } vec3i;

inline double pow_2(double x) { return x*x; }
inline double pow_3(double x) { return x*x*x; }
inline double pow_4(double x) { return x*x*x*x; }

/*
 * Returns length of solution vector:
 */
inline int get_length() {
    return (parameters.NDIMS == 3) ? (parameters.N * parameters.N * parameters.N) : (parameters.N * parameters.N);
}

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
extern float tau;                   // Dimensionless program time variable (in conformal time).
extern float T_initial;             // Initial temperature in units of f_a. Defined when H ~ f_a
extern float reduced_planck_mass;   // Reduced Planck mass in GeV normalised by the axion decay constant f_a: M_planck = 1 / sqrt(8*pi*G) / f_a
extern float m_saxion_initial;      // Initial value of saxion mass in units of f_a: m_saxion = sqrt(lambda) * f_a / f_a
extern float g_star;                // Relativistic degrees of freedom: 
extern float m_eff_squared;         // Effective mass of the PQ potential: m_eff^2 = lambda ( T^2/3 - fa^2 )
extern float light_crossing_time;   // Light crossing time: approximate time for light to travel one Hubble volume.
void  set_physics_variables();
float physical_time();
float scale_factor();
float hubble_parameter();
float temperature();
float string_tension();
float meff_squared();

// init.cpp
void gaussian_thermal(dtype *phi1, dtype *phi2, dtype *phidot1, dtype *phidot2);

// integrate.cpp
void  build_coefficient_matrix(sparse_matrix_t *handle, int NDIMS, int N);
dtype laplacian2D(dtype *phi, int i, int j, float dx, int N);
dtype laplacian3D(dtype *phi, int i, int j, int k, float dx, int N);
void  vvsl_field_rescaled(all_data data);
void  vvsl_hamiltonian_form(all_data data);

// string_finding.cpp
int cores2(dtype *axion, std::vector <vec2i> &s);
int cores3(dtype *axion, std::vector <vec3i> &s);

// mkl_wrapper.cpp
sparse_status_t mkl_wrapper_sparse_create_coo (sparse_matrix_t *A,
                                               const sparse_index_base_t indexing,
                                               const MKL_INT rows,
                                               const MKL_INT cols,
                                               const MKL_INT nnz,
                                               MKL_INT *row_indx,
                                               MKL_INT * col_indx,
                                               dtype *values);
sparse_status_t mkl_wrapper_sparse_mv (const sparse_operation_t operation,
                                       const dtype alpha,
                                       const sparse_matrix_t A,
                                       const struct matrix_descr descr,
                                       const dtype *x,
                                       const dtype beta,
                                       dtype *y);
void mkl_axpy (const MKL_INT n, const dtype a, const dtype *x, const MKL_INT incx, dtype *y, const MKL_INT incy);
void mkl_copy (const MKL_INT n, const dtype *x, const MKL_INT incx, dtype *y, const MKL_INT incy);
int  mkl_v_rng_gaussian(MKL_INT method, VSLStreamStatePtr stream, MKL_INT n, dtype *r, dtype a, dtype sigma);

// fileio.cpp
extern FILE *fp_main_output, *fp_time_series, *fp_snapshot_timings;
void read_field_data(const char *filepath, dtype *data, int length);
void save_data(char *file_name, dtype *data, int length);
void save_strings2(char *file_name, std::vector <vec2i> *v);
void save_strings3(char *file_name, std::vector <vec3i> *v);
void open_output_filestreams();
void close_output_filestreams();
