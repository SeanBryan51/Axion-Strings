#pragma once

#include "common/common.hpp"

#define REFINEMENT_FACTOR 2

#define BUFFER_STENCIL 4

/**
 * Note: to access elements a[i,j] = a[i + b_size * j] where i, j = 0, 1, ..., b_size - stencil - 1 where stencil will often be 2 when 
 * including a buffer.
 */
typedef struct block_data {
    int index_global; // "global index": index where memory starts for the buffer (including buffers).
    int index_sv;     // "solution vector index": index where the actual solution vector starts in the array. When there is no buffer, index_sv = 0.
    int size;         // "block size": size of square grid (including the buffer), i.e. N.
    int has_buffer;
    vec2i origin_global; // coordinate of origin in global simulation domain (the start of the solution vector) in units of space_step / pow(REFINEMENT_FACTOR, level)
} block_data;

/**
 * 
 * Solution vectors contain data for all blocks/patches defined on the level.
 * The information for each block in the array is specified by the b_data vector.
 * 
 * Note: assume grids are square for simplicity
 */
typedef struct level_data {
    int length;  // length of aggregate solution vector
    int tau_int; // tau in "integer units" to track time across refinement levels. tau = tau_initial + tau_int * (time_step / pow(REFINEMENT_FACTOR, level))

    // TODO: solve unnecessary buffer memory usage in solution vectors

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

    data_t *phi1_prev;  // store phi_1 field values from previous time step (use for time interpolation)
    data_t *phi2_prev;  // store phi_2 field values from previous time step

    std::vector<block_data> b_data; // Note: number of total blocks in level is given by b_data.size() 

    // worry about coefficient matrix later:
    // sparse_matrix_t coefficient_matrix;

} level_data;

/**
 * Block specifications for refinement.
 */
typedef struct block_spec_t {
    vec2i coord;
    int size;
} block_spec_t;

/**
 * Return type for interpolation of field quantities / averaging.
 */
typedef struct field_values {
    data_t phi1, phi2, phidot1, phidot2, ker1_curr, ker2_curr, ker1_next, ker2_next, phi1_prev, phi2_prev;
} field_values;

/**
 * Return type for function : coordinate_global_to_local()
 */
typedef struct global_to_local_return_t {
    vec2i local_coordinate;
    vec2i offsets;
} global_to_local_return_t;

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

/**
 * Converts local solution vector coordinates to global coordinates.
 */
inline vec2i coordinate_global(int i, int j, block_data b) {
    return (vec2i) {
        .x = b.origin_global.x + i,
        .y = b.origin_global.y + j
    };
}

/**
 * Converts global coordinates @param global_coordinate defined on level specified by @param level_in to the local solution vector
 * coordinates of block @param b which lives on level @param level_out.
 */
inline global_to_local_return_t coordinate_global_to_local(int level_in, vec2i global_coordinate, int level_out, block_data b) {

    global_to_local_return_t ret;
    int conversion_factor = ((int) pow(REFINEMENT_FACTOR, level_in - level_out));

    // local coordinate in units corresponding to input level.
    vec2i local_coord = {
        .x = global_coordinate.x - b.origin_global.x * conversion_factor, // need to convert block coordinate to same units as the input coordinate.
        .y = global_coordinate.y - b.origin_global.y * conversion_factor
    };

    // local coordinate in units corresponding to output level.
    ret.local_coordinate = (vec2i) {
        .x = local_coord.x / conversion_factor,
        .y = local_coord.y / conversion_factor
    };

    // Note: when offsets.x(y) = 0, the coordinate lands perfectly on a grid point.
    // When offsets.x(y) = some integer k where 0 < k < conversion_factor, the coordinate
    // is in between grid points and must be interpolated.
    ret.offsets = (vec2i) {
        .x = local_coord.x % conversion_factor,
        .y = local_coord.y % conversion_factor
    };

    return ret;
}

/**
 * Computes bilinear interpolation between four points.
 */
inline data_t interpolate2(data_t *field, int i, int j, int x_offset, int y_offset, int N, int starting_offset, data_t conversion_factor) {

    data_t field_00 = field[offset2(i, j, N, starting_offset)];
    data_t field_10 = field[offset2(i+1, j, N, starting_offset)];
    data_t field_01 = field[offset2(i, j+1, N, starting_offset)];
    data_t field_11 = field[offset2(i+1, j+1, N, starting_offset)];

    return (1.0f - x_offset / conversion_factor) * (1.0f - y_offset / conversion_factor) * field_00 + x_offset / conversion_factor * (1.0f - y_offset / conversion_factor) * field_10 + (1.0f - x_offset / conversion_factor) * y_offset / conversion_factor * field_01 + x_offset * y_offset / pow_2(conversion_factor) * field_11;
}

// amr_integrate.cpp
void evolve_level(std::vector<level_data*> hierarchy, int level);
void integrate_level(std::vector<level_data*> hierarchy, int level);
void reflux();

// amr_point_clustering.cpp
void gen_refinement_blocks(std::vector<block_spec_t> &to_refine, int *flagged, block_data b);

// amr_helper.cpp
field_values interpolate_from_hierarchy(std::vector<level_data*> hierarchy, int level_input, vec2i coord_input);
void fill_block_buffer(std::vector<level_data*> hierarchy, int level, block_data b);
void average_down(std::vector<level_data*> hierarchy, int level);
void regrid(std::vector<level_data*> hierarchy, int level, std::vector<std::vector<block_spec_t>> b_specs);
