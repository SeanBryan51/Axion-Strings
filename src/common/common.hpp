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

#include "amr/amr_interface.hpp"
#include "standard/s_interface.hpp"

#ifdef USE_DOUBLE_PRECISION
typedef double data_t;
#else
typedef float data_t;
#endif

#define MAX_LEN 200

// Macro for periodic boundary conditions:
#define periodic(i,N) (((i) >= 0) ? (i) % (N) : (N) - (-(i) % (N)))

extern struct parameters {

    // User defined parameters to be read from 
    // parameter file:

    float lambdaPRS;
    int   NDIMS;
    int   N;
    float space_step;
    float time_step;
    int   stencil_setting;
    unsigned int seed;

    int  write_output_file;
    char output_file_path[MAX_LEN];

    int  save_snapshots;
    int  n_snapshots;
    char output_directory[MAX_LEN];
    int  save_fields;
    int  save_strings;

    int  sample_time_series;
    int  n_samples;
    char ts_output_path[MAX_LEN];

    int sample_strings;
    int sample_background;

    int thr;

    int   enable_amr;
    float refinement_threshold;
    int   init_from_snapshot;
    float tau_initial;

} parameters;

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
extern data_t tau;                  // Dimensionless program time variable (in conformal time).
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
void gaussian_thermal(data_t *phi1, data_t *phi2, data_t *phidot1, data_t *phidot2);

// string_finding.cpp
int cores2(data_t *axion, std::vector <vec2i> &s);
int cores3(data_t *axion, std::vector <vec3i> &s);

// mkl_wrapper.cpp
sparse_status_t mkl_wrapper_sparse_create_coo (sparse_matrix_t *A,
                                               const sparse_index_base_t indexing,
                                               const MKL_INT rows,
                                               const MKL_INT cols,
                                               const MKL_INT nnz,
                                               MKL_INT *row_indx,
                                               MKL_INT * col_indx,
                                               data_t *values);
sparse_status_t mkl_wrapper_sparse_mv (const sparse_operation_t operation,
                                       const data_t alpha,
                                       const sparse_matrix_t A,
                                       const struct matrix_descr descr,
                                       const data_t *x,
                                       const data_t beta,
                                       data_t *y);
void mkl_axpy (const MKL_INT n, const data_t a, const data_t *x, const MKL_INT incx, data_t *y, const MKL_INT incy);
void mkl_copy (const MKL_INT n, const data_t *x, const MKL_INT incx, data_t *y, const MKL_INT incy);
int  mkl_v_rng_gaussian(MKL_INT method, VSLStreamStatePtr stream, MKL_INT n, data_t *r, data_t a, data_t sigma);

// utils/fileio.cpp
extern FILE *fp_main_output, *fp_time_series, *fp_snapshot_timings;
void fio_open_output_filestreams();
void fio_close_output_filestreams();
void fio_read_field_data(char *file_name, data_t *data, int length);
void fio_save_field_data(char *file_name, data_t *data, int length);
void fio_save_strings2(char *file_name, std::vector <vec2i> *v);
void fio_save_strings3(char *file_name, std::vector <vec3i> *v);
void fio_save_flagged_data(char *file_name, int *data, int length);

// utils/read_parameters.cpp
void read_parameter_file(char *fname);
