#pragma once

#include "common/common.hpp"

/*
 * Solution vectors contain data for all blocks/patches defined on the level.
 * The location of the solution vector for each patch in the array is specified by
 * the b_index array.
 */
typedef struct level_data {
    // assume grids are square for simplicity
    int size;      // size of current solution vector

    data_t *phi1;       // phi_1 field values
    data_t *phi2;       // phi_2 field values
    data_t *phidot1;    // phi_1 time derivative
    data_t *phidot2;    // phi_2 time derivative
    data_t *ker1_curr;  // current kernel for phi_1 equation of motion
    data_t *ker2_curr;  // current kernel for phi_2 equation of motion
    data_t *ker1_next;  // next kernel for phi_1 equation of motion
    data_t *ker2_next;  // next kernel for phi_2 equation of motion
    data_t *axion;      // axion field values
    data_t *saxion;     // saxion field values

    std::vector<int> b_index; // "block index": b_index[n] gives the starting index in the solution vector for the nth block.
    std::vector<int> b_size;  // "block size": size of square grid.

    // worry about coefficient matrix later:
    // sparse_matrix_t coefficient_matrix;

} level_data;
