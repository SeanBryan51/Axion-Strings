#include "amr_internal.hpp"

/**
 * Given a global coordinate @param coord_input defined on the level @param level_input, the function returns
 * a set of interpolated field values from the hierarchy.
 * 
 * Cases: the field values are interpolated from either
 * 1. fine-fine (same level) (can also include periodic boundary conditions)
 * 2. coarse-fine (from level below)
 * 3. physical boundary conditions (periodic boundary conditions on root_level)
 */
field_values interpolate_from_hierarchy(std::vector<level_data*> hierarchy, int level_input, vec2i coord_input) {
    field_values ret;

    // start with the given level, if no blocks correspond to the given coordinate, check the level below.
    for (int l = level_input; l >= 0; l--) {
        level_data data = *hierarchy[l];

        // start checking blocks if global coordinate is in that block (i.e. the global coord corresponds to a valid index of the solution vector)
        for (block_data b : data.b_data) {
            int b_sv_size = (b.has_buffer) ? b.size - BUFFER_STENCIL : b.size;

            global_to_local_return_t c = coordinate_global_to_local(level_input, coord_input, l, b);
            int i = c.local_coordinate.x, j = c.local_coordinate.y, x_offset = c.offsets.x, y_offset = c.offsets.y;

            if (i >= 0 && i < b_sv_size && j >= 0 && j < b_sv_size) {
                // the block contains the coordinate:

                data_t conversion_factor = pow(REFINEMENT_FACTOR, level_input - l);
                int t_offset = hierarchy[level_input]->tau_int % ((int) conversion_factor);

                if (t_offset == 0)
                    assert(hierarchy[level_input]->tau_int == data.tau_int * ((int) conversion_factor)); // sanity check

                // compute bilinear interpolation between four points:

                data_t phi1_prev = interpolate2(data.phi1_prev, i, j, x_offset, y_offset, b.size, b.index_global + b.index_sv, conversion_factor);
                data_t phi1_curr = interpolate2(data.phi1, i, j, x_offset, y_offset, b.size, b.index_global + b.index_sv, conversion_factor);
                ret.phi1 = (1.0f - t_offset / conversion_factor) * phi1_prev + t_offset / conversion_factor * phi1_curr;

                data_t phi2_prev = interpolate2(data.phi2_prev, i, j, x_offset, y_offset, b.size, b.index_global + b.index_sv, conversion_factor);
                data_t phi2_curr = interpolate2(data.phi2, i, j, x_offset, y_offset, b.size, b.index_global + b.index_sv, conversion_factor);
                ret.phi2 = (1.0f - t_offset / conversion_factor) * phi2_prev + t_offset / conversion_factor * phi2_curr;

                return ret; // return as soon as we have found the value.
            }
        }
    }

    // if the code reaches this point, the coordinate couldn't be found in the hierarchy?
    assert(0);

    return ret;
}

/**
 * Fills the buffer of the given block (which lives on the specified level) with appropriate boundary conditions.
 */
void fill_block_buffer(std::vector<level_data*> hierarchy, int level, block_data b) {

    level_data data = *hierarchy[level];
    data_t *phi1 = &data.phi1[b.index_global];
    data_t *phi2 = &data.phi2[b.index_global];

    // offset origin global to the start of the buffer rather than start of solution vector.
    // Note: this doesn't modify the actual buffer data as the buffer was passed in by value.
    b.origin_global.x -= BUFFER_STENCIL / 2;
    b.origin_global.y -= BUFFER_STENCIL / 2;

    for (int j = 0; j < BUFFER_STENCIL / 2; j++) {
        for (int i = 0; i < b.size; i++) {

            field_values bc;
            vec2i global_coord;
            int domain_size = hierarchy[0]->b_data[0].size * ((int) pow(REFINEMENT_FACTOR, level));

            // top:
            int top_indx = i + b.size * j;
            global_coord = {
                .x = periodic(b.origin_global.x + i, domain_size),
                .y = periodic(b.origin_global.y + j, domain_size)
            };
            bc = interpolate_from_hierarchy(hierarchy, level, global_coord);
            phi1[top_indx] = bc.phi1;
            phi2[top_indx] = bc.phi2;

            // bottom:
            int bot_indx = i + b.size * (b.size - 1 - j);
            global_coord = {
                .x = periodic(b.origin_global.x + i, domain_size),
                .y = periodic(b.origin_global.y + (b.size - 1 - j), domain_size)
            };
            bc = interpolate_from_hierarchy(hierarchy, level, global_coord);
            phi1[bot_indx] = bc.phi1;
            phi2[bot_indx] = bc.phi2;

            // left:
            int left_indx = j + b.size * i;
            global_coord = {
                .x = periodic(b.origin_global.x + j, domain_size),
                .y = periodic(b.origin_global.y + i, domain_size)
            };
            bc = interpolate_from_hierarchy(hierarchy, level, global_coord);
            phi1[left_indx] = bc.phi1;
            phi2[left_indx] = bc.phi2;

            // right:
            int right_indx = (b.size - 1 - j) + b.size * i;
            global_coord = {
                .x = periodic(b.origin_global.x + (b.size - 1 - j), domain_size),
                .y = periodic(b.origin_global.y + i, domain_size)
            };
            bc = interpolate_from_hierarchy(hierarchy, level, global_coord);
            phi1[right_indx] = bc.phi1;
            phi2[right_indx] = bc.phi2;

        }
    }
}

/**
 * Sets the values of the specified level @param level to the coarser grid underneath.
 * Assumes hierarchy is properly nested.
 */
void average_down(std::vector<level_data*> hierarchy, int level) {

    assert(level > 0);

    level_data data_fine = *hierarchy[level];
    level_data data_coarse = *hierarchy[level - 1];

    for (block_data b_fine : data_fine.b_data) {
        int b_fine_sv_size = (b_fine.has_buffer) ? b_fine.size - BUFFER_STENCIL : b_fine.size;

        int i_starting_index = b_fine.origin_global.x % REFINEMENT_FACTOR, i_inc = REFINEMENT_FACTOR; // We only want to iterate over elements that align with coarse grid.
        int j_starting_index = b_fine.origin_global.y % REFINEMENT_FACTOR, j_inc = REFINEMENT_FACTOR;
        #pragma omp parrallel for schedule(static) collapse(2)
        for (int i = i_starting_index; i < b_fine_sv_size; i += i_inc) {
            for (int j = j_starting_index; j < b_fine_sv_size; j += j_inc) {

                vec2i global_coord = coordinate_global(i, j, b_fine);

                for (block_data b_coarse : data_coarse.b_data) {
                    int b_coarse_sv_size = (b_coarse.has_buffer) ? b_coarse.size - BUFFER_STENCIL : b_coarse.size;

                    global_to_local_return_t ret = coordinate_global_to_local(level, global_coord, level - 1, b_coarse);

                    assert(ret.offsets.x == 0 && ret.offsets.y == 0); // Sanity check: we should be iterating over elements that align with the coarse grid.

                    if (ret.local_coordinate.x < 0 || ret.local_coordinate.x >= b_coarse_sv_size || ret.local_coordinate.y < 0 || ret.local_coordinate.y >= b_coarse_sv_size)
                        continue; // block does not contain the coordinate.

                    // Set coarse values to the fine.
                    int offset_fine = offset2(i, j, b_fine.size, b_fine.index_global + b_fine.index_sv);
                    int offset_coarse = offset2(ret.local_coordinate.x, ret.local_coordinate.y, b_coarse.size, b_coarse.index_global + b_coarse.index_sv);
                    data_coarse.phi1[offset_coarse] = data_fine.phi1[offset_fine];
                    data_coarse.phi2[offset_coarse] = data_fine.phi2[offset_fine];
                    data_coarse.phidot1[offset_coarse] = data_fine.phidot1[offset_fine];
                    data_coarse.phidot2[offset_coarse] = data_fine.phidot2[offset_fine];
                    data_coarse.ker1_curr[offset_coarse] = data_fine.ker1_curr[offset_fine];
                    data_coarse.ker2_curr[offset_coarse] = data_fine.ker2_curr[offset_fine];

                }
            }
        }

    }
}

void regrid(std::vector<level_data*> hierarchy, int level, std::vector<std::vector<block_spec_t>> b_specs) {
    // TODO: implement regridding
}