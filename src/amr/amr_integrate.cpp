#include "amr_internal.hpp"

/**
 * Implementation of the Berger-Collela time-stepping algorithm:
 * Function takes in a grid hierarchy and evolves the level
 * specified by level by one time step.
 */
void evolve_level(std::vector<level_data*> hierarchy, int level) {

    level_data data = *hierarchy[level];
    data_t dt_local = parameters.time_step / pow(REFINEMENT_FACTOR, level);

    integrate_level(hierarchy, level);

    std::vector<std::vector<block_spec_t>> b_specs;
    for (block_data &b : data.b_data) {
        std::vector<block_spec_t> to_refine = {};
        int *flagged = &data.flagged[b.index_global];
        // Fill b_specs with optimal boxes for refinement:
        gen_refinement_blocks(to_refine, flagged, b);
        b_specs.push_back(to_refine);
    }

    // This is going to be one hell of a function!
    regrid(hierarchy, level, b_specs);

    if (level + 1 < hierarchy.size()) {
        // There exists a finer level:

        // Perform subcycling:
        evolve_level(hierarchy, level + 1); // For a factor of 2 refinement between levels, we need to evolve the level twice:
        hierarchy[level + 1]->tau_int += 1;

        evolve_level(hierarchy, level + 1);
        hierarchy[level + 1]->tau_int += 1;

        assert(hierarchy[level + 1]->tau_int == (hierarchy[level]->tau_int + 1) * REFINEMENT_FACTOR); // Sanity check: times on both levels are in sync.

        reflux();
        average_down(hierarchy, level + 1); // set covered coarse cells to be the average of fine
    }

}

/**
 * Integrates the given level forward in time without knowing the fine level data and
 * checks the refinement condition after integrating fields.
 * 
 */
void integrate_level(std::vector<level_data*> hierarchy, int level) {

    level_data data = *hierarchy[level];

    float dx = parameters.space_step / pow(REFINEMENT_FACTOR, level);
    float dt = parameters.time_step / pow(REFINEMENT_FACTOR, level);

    for (block_data &b : data.b_data) {
        int min_index = (b.has_buffer) ? BUFFER_STENCIL / 2 : 0;
        int max_index = (b.has_buffer) ? b.size - BUFFER_STENCIL / 2 : b.size;

        // Save field values for interpolation:
        #pragma omp parrallel for schedule(static) collapse(2)
        for (int i = min_index; i < max_index; i++) {
            for (int j = min_index; j < max_index; j++) {
                int offset = offset2(i, j, b.size, b.index_global);
                data.phi1_prev[offset] = data.phi1[offset];
                data.phi2_prev[offset] = data.phi2[offset];
            }
        }
    }

    for (block_data &b : data.b_data) {
        int min_index = (b.has_buffer) ? BUFFER_STENCIL / 2 : 0;
        int max_index = (b.has_buffer) ? b.size - BUFFER_STENCIL / 2 : b.size;

        #pragma omp parrallel for schedule(static) collapse(2)
        for (int i = min_index; i < max_index; i++) {
            for (int j = min_index; j < max_index; j++) {
                int offset = offset2(i, j, b.size, b.index_global);
                data.phi1[offset] += dt * (data.phidot1[offset] + 0.5f * data.ker1_curr[offset] * dt);
                data.phi2[offset] += dt * (data.phidot2[offset] + 0.5f * data.ker2_curr[offset] * dt);
            }
        }
    }

    for (block_data &b : data.b_data) {
        if (b.has_buffer) fill_block_buffer(hierarchy, level, b);
        // debug
        for (int i = 0; i < b.size; i++) {
            for (int j = 0; j < b.size; j++) {
                int offset = offset2(i, j, b.size, b.index_global);
                assert(data.phi1[offset] != 0);
                assert(data.phi2[offset] != 0);
            }
        }
    }

    for (block_data &b : data.b_data) {
        int min_index = (b.has_buffer) ? BUFFER_STENCIL / 2 : 0;
        int max_index = (b.has_buffer) ? b.size - BUFFER_STENCIL / 2 : b.size;

        #pragma omp parallel for schedule(static) collapse(2)
        for (int i = min_index; i < max_index; i++) {
            for (int j = min_index; j < max_index; j++) {

                int offset = offset2(i, j, b.size, b.index_global);
                data_t lap1 = laplacian2(data.phi1, i, j, b.size, b.index_global) / pow_2(dx);
                data_t lap2 = laplacian2(data.phi2, i, j, b.size, b.index_global) / pow_2(dx);

                data_t tau_local = parameters.tau_initial + data.tau_int * dt;

                data.ker1_next[offset] = lap1 - parameters.lambda / pow(tau_local, 2.0f * parameters.enable_PRS) * data.phi1[offset] * (pow_2(data.phi1[offset]) + pow_2(data.phi2[offset]) - pow_2(tau_local) + pow_2(T_initial) / (3.0f));
                data.ker2_next[offset] = lap2 - parameters.lambda / pow(tau_local, 2.0f * parameters.enable_PRS) * data.phi2[offset] * (pow_2(data.phi1[offset]) + pow_2(data.phi2[offset]) - pow_2(tau_local) + pow_2(T_initial) / (3.0f));
            }
        }
    }

    for (block_data &b : data.b_data) {
        int min_index = (b.has_buffer) ? BUFFER_STENCIL / 2 : 0;
        int max_index = (b.has_buffer) ? b.size - BUFFER_STENCIL / 2 : b.size;

        #pragma omp parallel for schedule(static) collapse(2)
        for (int i = min_index; i < max_index; i++) {
            for (int j = min_index; j < max_index; j++) {
                int offset = offset2(i, j, b.size, b.index_global);
                data.phidot1[offset] += 0.5f * (data.ker1_curr[offset] + data.ker1_next[offset]) * dt;
                data.phidot2[offset] += 0.5f * (data.ker2_curr[offset] + data.ker2_next[offset]) * dt;
            }
        }

        #pragma omp parallel for schedule(static) collapse(2)
        for (int i = min_index; i < max_index; i++) {
            for (int j = min_index; j < max_index; j++) {
                int offset = offset2(i, j, b.size, b.index_global);
                data.ker1_curr[offset] = data.ker1_next[offset];
                data.ker2_curr[offset] = data.ker2_next[offset];
            }
        }
    }

    for (block_data &b : data.b_data) {
        int min_index = (b.has_buffer) ? BUFFER_STENCIL / 2 : 0;
        int max_index = (b.has_buffer) ? b.size - BUFFER_STENCIL / 2 : b.size;

        // flag points that meet refinement criterion:
        #pragma omp parallel for schedule(static) collapse(2)
        for (int i = min_index; i < max_index; i++) {
            for (int j = min_index; j < max_index; j++) {
                int offset = offset2(i, j, b.size, b.index_global);

                // Note: don't have to divide gradient by dx^2 for refinement criterion.
                data_t grad_sq1 = gradient_squared2(data.phi1, i, j, b.size, b.index_global);
                data_t grad_sq2 = gradient_squared2(data.phi2, i, j, b.size, b.index_global);

                data.flagged[offset] = (sqrt(grad_sq1 + grad_sq2) > parameters.refinement_threshold);
            }
        }

    }

}

void reflux() {
    // TODO: implement reflux
}
