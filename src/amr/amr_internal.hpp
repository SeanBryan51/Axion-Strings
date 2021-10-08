#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#include <omp.h>

#include "mkl.h"
#include "mkl_vsl.h"
#include "mkl_spblas.h"

#include "../parameters.h"

#ifdef USE_DOUBLE_PRECISION
typedef double data_t;
#else
typedef float data_t;
#endif

typedef unsigned int uint32;

typedef struct _level_data {
    data_t *phi1; // phi_1 field values
    data_t *phi2; // phi_2 field values
    data_t *phidot1; // phi_1 time derivative
    data_t *phidot2; // phi_2 time derivative
    data_t *ker1_curr; // current kernel for phi_1 equation of motion
    data_t *ker2_curr; // current kernel for phi_2 equation of motion
    data_t *ker1_next; // next kernel for phi_1 equation of motion
    data_t *ker2_next; // next kernel for phi_2 equation of motion

    data_t *axion; // axion field values
    data_t *saxion; // saxion field values



    sparse_matrix_t coefficient_matrix;

} level_data;

struct Hierarchy {
    int num_levels;
    Level *levels; // (Array of pointers to structs)
};

struct Level {
    int num_blocks;
    float dx, dtau;
    Block *blocks; // (Array of pointers to structs)
};

struct Block {
    // For 2D case: phi1(x,y) = phi1[x + y * width]
    int width, height;
    float xmin, xmax;
    float ymin, ymax;
    float *phi1;
    float *phi2;
    float *phidot1;
    float *phidot2;
};
