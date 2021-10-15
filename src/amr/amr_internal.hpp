#pragma once

#include "common/common.hpp"

// Note: to access elements a[i,j] = a[i + b_size * j] where i, j = 0, 1, ..., b_size - stencil - 1 where stencil will often be 2 when including a buffer.
typedef struct block_data {
    int index_global; // "global index": index where memory starts for the buffer (including buffers).
    int index_sv;     // "solution vector index": index where the actual solution vector starts in the array. When there is no buffer, index_sv = 0.
    int size;         // "block size": size of square grid (including the buffer), i.e. N.
    int has_buffer;
} block_data;

/**
 * 
 * Solution vectors contain data for all blocks/patches defined on the level.
 * The information for each block in the array is specified by the b_data vector.
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
inline data_t laplacian2(data_t *field, int i, int j, int N, int so = 0) {
    return field[offset2(i+1,j,N,so)] + field[offset2(i,j+1,N,so)] - 4.0f*field[offset2(i,j,N,so)] + field[offset2(i-1,j,N,so)] + field[offset2(i,j-1,N,so)];
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
inline data_t gradient_squared2(data_t *field, int i, int j, int N, int so = 0) {
    data_t grad_x = field[offset2(i+1,j,N,so)] - 2.0f*field[offset2(i,j,N,so)] + field[offset2(i-1,j,N,so)];
    data_t grad_y = field[offset2(i,j+1,N,so)] - 2.0f*field[offset2(i,j,N,so)] + field[offset2(i,j-1,N,so)];
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
void regrid(std::vector<level_data> hierarchy, std::vector<vec2i> block_coords, std::vector<int> block_size, int level);
void reflux();
void average_down();

// amr_point_clustering.cpp
void gen_refinement_blocks(std::vector<vec2i> &block_coords, std::vector<int> &block_size, std::vector<level_data> hierarchy, int level);
