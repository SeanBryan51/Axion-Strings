#include "amr_internal.hpp"

#define REFINEMENT_FACTOR 2

/*
 * Implementation of the Berger-Collela time-stepping algorithm:
 * Function takes in a grid hierarchy and evolves the level
 * specified by level by one time step.
 */
void evolve_level(std::vector<level_data> hierarchy, int level, data_t tau_local) {

    data_t dt_local = parameters.time_step / pow(REFINEMENT_FACTOR, level);

    integrate_level(hierarchy, level, tau_local);

    // Random notes:
    // - Refine at a specific timestep (not everytime as it is computationally expensive?).
    // Solution: integrate_level() should check the refinement condition after integrating fields.
    // - Buffers allow features to move within a patch between refinement periods. Can also gamble by using
    // a very accurate refinement criterion so we can be sure we don't lose resolution of features in between 
    // refinement periods?
    // Solution: for now just check after every integration.
    // - Actually this should be executed almost every time, for example what if the current level no longer satifies the refinement
    // condition and all higher levels need to be freed?

    // assume 2D for now:
    std::vector<vec2i> block_coords;
    std::vector<int> block_size;

    // Populate arguments block_coords and block_size with optimal boxes for refinement:
    gen_refinement_blocks(block_coords, block_size, hierarchy, level);

    // This is going to be one hell of a function!
    regrid(hierarchy, block_coords, block_size, level);

    if (level + 1 < hierarchy.size()) {
        // There exists a finer level:

        // Perform subcycling:
        // For a factor of 2 refinement between levels, we need to evolve the level twice:
        evolve_level(hierarchy, level + 1, tau_local);
        tau_local += dt_local;
        evolve_level(hierarchy, level + 1, tau_local);
        tau_local += dt_local;

        reflux();
        average_down(); // set covered coarse cells to be the average of fine
    }

}

/*
 * Integrates the given level forward in time without knowing the fine level data
 */
void integrate_level(std::vector<level_data> hierarchy, int level, data_t tau_local) {

    level_data data = hierarchy[level];
    // int NDIMS = parameters.NDIMS;
    // int length = data.size;

    float dx = parameters.space_step / pow(REFINEMENT_FACTOR, level);
    float dt = parameters.time_step / pow(REFINEMENT_FACTOR, level);

    // TODO: this is so broken (indexing backwards to access buffer)

    int n_blocks = data.b_data.size();
    for (int b_id = 0; b_id < n_blocks; b_id++) {
        int b_size = data.b_data[b_id].size;
        int b_ij_size = (data.b_data[b_id].has_buffer) ? b_size - 2 : b_size;
        // offset at which i, j = b_ij_size - stencil - 1
        int b_max_offset = (b_ij_size - 1) + b_size * (b_ij_size - 1); // assume 2D for now.

        data_t *phi1      = &data.phi1[data.b_data[b_id].index];
        data_t *phi2      = &data.phi2[data.b_data[b_id].index];
        data_t *phidot1   = &data.phidot1[data.b_data[b_id].index];
        data_t *phidot2   = &data.phidot2[data.b_data[b_id].index];
        data_t *ker1_curr = &data.ker1_curr[data.b_data[b_id].index];
        data_t *ker2_curr = &data.ker2_curr[data.b_data[b_id].index];
        data_t *ker1_next = &data.ker1_next[data.b_data[b_id].index];
        data_t *ker2_next = &data.ker2_next[data.b_data[b_id].index];

        int *flagged = &data.flagged[data.b_data[b_id].index];

        #pragma omp parallel for schedule(static)
        for (int l = 0; l <= b_max_offset; l++) {
            int i, j;
            coordinate2(&i, &j, l, b_size);
            // Check if current index is a part of the buffer:
            if (!(i >= 0 && i < b_ij_size && j >= 0 && j < b_ij_size))
                continue;

            phi1[l] += dt * (phidot1[l] + 0.5f * ker1_curr[l] * dt);
            phi2[l] += dt * (phidot2[l] + 0.5f * ker2_curr[l] * dt);
        }

        // tau += dt; // Do this only on the root level

        #pragma omp parallel for schedule(static)
        for (int l = 0; l <= b_max_offset; l++) {
            int i, j;
            coordinate2(&i, &j, l, b_size);
            // Check current index is a part of the buffer:
            if (!(i >= 0 && i < b_ij_size && j >= 0 && j < b_ij_size))
                continue;

            data_t lap1 = laplacian2(phi1, i, j, b_size) / pow_2(dx);
            data_t lap2 = laplacian2(phi2, i, j, b_size) / pow_2(dx);
            // ker1_next[l] = lap1 - 1.0f / pow_2(tau_local) * parameters.lambdaPRS * phi1[l] * (pow_2(phi1[l]) + pow_2(phi2[l]) - pow_2(tau_local) + pow_2(T_initial) / (3.0f)); // prs
            // ker2_next[l] = lap2 - 1.0f / pow_2(tau_local) * parameters.lambdaPRS * phi2[l] * (pow_2(phi1[l]) + pow_2(phi2[l]) - pow_2(tau_local) + pow_2(T_initial) / (3.0f)); // prs
            ker1_next[l] = lap1 - parameters.lambdaPRS * phi1[l] * (pow_2(phi1[l]) + pow_2(phi2[l]) - pow_2(tau_local) + pow_2(T_initial) / (3.0f)); // physical
            ker2_next[l] = lap2 - parameters.lambdaPRS * phi2[l] * (pow_2(phi1[l]) + pow_2(phi2[l]) - pow_2(tau_local) + pow_2(T_initial) / (3.0f)); // physical
        }

        #pragma omp parallel for schedule(static)
        for (int l = 0; l <= b_max_offset; l++) {
            int i, j;
            coordinate2(&i, &j, l, b_size);
            // Check current index is a part of the buffer:
            if (!(i >= 0 && i < b_ij_size && j >= 0 && j < b_ij_size))
                continue;

            phidot1[l] += 0.5f * (ker1_curr[l] + ker1_next[l]) * dt;
            phidot2[l] += 0.5f * (ker2_curr[l] + ker2_next[l]) * dt;
        }

        #pragma omp parallel for schedule(static)
        for (int l = 0; l <= b_max_offset; l++) {
            ker1_curr[l] = ker1_next[l];
            ker2_curr[l] = ker2_next[l];
        }

        // flag points that meet refinement criterion:
        #pragma omp parallel for schedule(static)
        for (int l = 0; l <= b_max_offset; l++) {
            int i, j;
            coordinate2(&i, &j, l, b_size);
            // Check current index is a part of the buffer:
            if (!(i >= 0 && i < b_ij_size && j >= 0 && j < b_ij_size))
                continue;

            // Note: don't have to divide gradient by dx^2 for refinement criterion.
            data_t grad_sq1 = gradient_squared2(phi1, i, j, b_size);
            data_t grad_sq2 = gradient_squared2(phi2, i, j, b_size);

            flagged[l] = (sqrt(grad_sq1 + grad_sq2) > parameters.refinement_threshold);
        }
    }


#if 0
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < b_length && level == 0; i++) {
        // Only makes sense on the root level where boundary conditions are periodic 
        // and are not set by the block buffer.
        data_t lap1, lap2;
        if (NDIMS == 2) {
            int x, y;
            coordinate2(&x, &y, i, N);
            lap1 = laplacian2(data.phi1, x, y, N) / pow_2(dx);
            lap2 = laplacian2(data.phi2, x, y, N) / pow_2(dx);
        } else {
            int x, y, z;
            coordinate3(&x, &y, &z, i, N);
            lap1 = laplacian3(data.phi1, x, y, z, N) / pow_2(dx);
            lap2 = laplacian3(data.phi2, x, y, z, N) / pow_2(dx);
        }
        data.ker1_next[i] = lap1 - 1.0f / pow_2(tau_local) * parameters.lambdaPRS * data.phi1[i] * (pow_2(data.phi1[i]) + pow_2(data.phi2[i]) - pow_2(tau_local) + pow_2(T_initial) / (3.0f));
        data.ker2_next[i] = lap2 - 1.0f / pow_2(tau_local) * parameters.lambdaPRS * data.phi2[i] * (pow_2(data.phi1[i]) + pow_2(data.phi2[i]) - pow_2(tau_local) + pow_2(T_initial) / (3.0f));
    }
#endif
#if 0
    // flag points that meet refinement criterion:
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < length; i++) {
        // Note: don't have to divide gradient by dx^2 for refinement criterion.
        data_t grad_sq1, grad_sq2;
        if (NDIMS == 2) {
            int x, y;
            coordinate2(&x, &y, i, N);
            grad_sq1 = gradient_squared2(data.phi1, x, y, N);
            grad_sq2 = gradient_squared2(data.phi2, x, y, N);
        } else {
            int x, y, z;
            coordinate3(&x, &y, &z, i, N);
            grad_sq1 = gradient_squared3(data.phi1, x, y, z, N);
            grad_sq2 = gradient_squared3(data.phi2, x, y, z, N);
        }

        float threshold = 0.3f;
        if (sqrt(grad_sq1 + grad_sq2) > threshold) {
            data.flagged[i] = 1;
        }
    }
#endif

}
