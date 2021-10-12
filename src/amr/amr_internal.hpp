#pragma once

#include "common/common.hpp"

// Note: to access elements a[i,j] = a[i + b_size * j] where i, j = 0, 1, ..., b_size - stencil - 1 where stencil will often be 2 when including a buffer.
typedef struct block_data {
    // TODO: this is not how you should do it!
    //       treat the buffer index as the 'global index'
    //       then have a 'relative index' that points to the start of the solution vector relative to buffer index.
    int index; // "block index": index gives the starting index in the solution vector for the nth block. Note the starting index does not start at the beggining of the buffer boundary conditions, but starts at the actual solution vector.
    int buffer_index; // "block buffer index": buffer index gives the starting index for the buffer boundary conditions of the nth block. When there is no buffer, buffer_index = index.
    int size; // "block size": size of square grid (including the buffer), i.e. N.
    int has_buffer;
} block_data;

/*
 * Solution vectors contain data for all blocks/patches defined on the level.
 * The location of the solution vector for each patch in the array is specified by
 * the b_index array.
 * 
 * Note: assume grids are square for simplicity
 */
typedef struct level_data {
    int length; // length of current solution vector

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

    int *flagged; // boolean vector used to flag grid points for refinement.

    std::vector<block_data> b_data; // Note: number of total blocks in level is given by b_data.size() 

    // worry about coefficient matrix later:
    // sparse_matrix_t coefficient_matrix;

} level_data;

/*
 * Note: remember to divide by the square of the lattice spacing.
 */
inline data_t laplacian2(data_t *field, int i, int j, int N) {
    return field[offset2(i+1,j,N)] + field[offset2(i,j+1,N)] - 4.0f*field[offset2(i,j,N)] + field[offset2(i-1,j,N)] + field[offset2(i,j-1,N)];
}

/*
 * Note: remember to divide by the square of the lattice spacing.
 */
inline data_t laplacian3(data_t *field, int i, int j, int k, int N) {
    return field[offset3(i+1,j,k,N)] + field[offset3(i,j+1,k,N)] + field[offset3(i,j,k+1,N)] - 6.0f*field[offset3(i,j,k,N)] + field[offset3(i-1,j,k,N)] + field[offset3(i,j-1,k,N)] + field[offset3(i,j,k-1,N)];
}

/*
 * Note: remember to divide by the square of the lattice spacing.
 */
inline data_t gradient_squared2(data_t *field, int i, int j, int N) {
    data_t grad_x = field[offset2(i+1,j,N)] - 2.0f*field[offset2(i,j,N)] + field[offset2(i-1,j,N)];
    data_t grad_y = field[offset2(i,j+1,N)] - 2.0f*field[offset2(i,j,N)] + field[offset2(i,j-1,N)];
    return pow_2(grad_x) + pow_2(grad_y);
}

/*
 * Note: remember to divide by the square of the lattice spacing.
 */
inline data_t gradient_squared3(data_t *field, int i, int j, int k, int N) {
    data_t grad_x = field[offset3(i+1,j,k,N)] - 2.0f*field[offset3(i,j,k,N)] + field[offset3(i-1,j,k,N)];
    data_t grad_y = field[offset3(i,j+1,k,N)] - 2.0f*field[offset3(i,j,k,N)] + field[offset3(i,j-1,k,N)];
    data_t grad_z = field[offset3(i,j,k+1,N)] - 2.0f*field[offset3(i,j,k,N)] + field[offset3(i,j,k-1,N)];
    return pow_2(grad_x) + pow_2(grad_y) + pow_2(grad_z);
}

// amr_integrate.cpp
void evolve_level(std::vector<level_data> hierarchy, int level, data_t tau_local);
void integrate_level(std::vector<level_data> hierarchy, int level, data_t tau_local);

// amr_point_clustering.cpp
void gen_refinement_blocks(std::vector<vec2i> &block_coords, std::vector<int> &block_size, std::vector<level_data> hierarchy, int level);

// amr_hierarchy.cpp
void regrid(std::vector<level_data> hierarchy, std::vector<vec2i> block_coords, std::vector<int> block_size, int level);
void reflux();
void average_down();
