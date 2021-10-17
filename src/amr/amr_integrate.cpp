#include "amr_internal.hpp"

#define REFINEMENT_FACTOR 2

/**
 * Implementation of the Berger-Collela time-stepping algorithm:
 * Function takes in a grid hierarchy and evolves the level
 * specified by level by one time step.
 */
void evolve_level(std::vector<level_data*> hierarchy, int level, data_t tau_local) {

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

    // Fill b_specs with optimal boxes for refinement:
    std::vector<std::vector<block_specs>> b_specs;
    level_data data = *hierarchy[level];
    for (block_data &b : data.b_data) {
        std::vector<block_specs> to_refine = {};
        int *flagged = &data.flagged[b.index_global];
        gen_refinement_blocks(to_refine, flagged, b);
        b_specs.push_back(to_refine);
    }

    // This is going to be one hell of a function!
    regrid(hierarchy, level, b_specs);

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

/**
 * Integrates the given level forward in time without knowing the fine level data
 * 
 * @returns 1 if points have been flagged for refinement, 0 otherwise.
 */
void integrate_level(std::vector<level_data*> hierarchy, int level, data_t tau_local) {

    level_data data = *hierarchy[level];

    float dx = parameters.space_step / pow(REFINEMENT_FACTOR, level);
    float dt = parameters.time_step / pow(REFINEMENT_FACTOR, level);

    for (block_data &b : data.b_data) {
        int b_sv_size = (b.has_buffer) ? b.size - BUFFER_STENCIL : b.size;

        data_t *phi1      = &data.phi1[b.index_global];
        data_t *phi2      = &data.phi2[b.index_global];
        data_t *phidot1   = &data.phidot1[b.index_global];
        data_t *phidot2   = &data.phidot2[b.index_global];
        data_t *ker1_curr = &data.ker1_curr[b.index_global];
        data_t *ker2_curr = &data.ker2_curr[b.index_global];

        #pragma omp parrallel for schedule(static) collapse(2)
        for (int i = 0; i < b_sv_size; i++) {
            for (int j = 0; j < b_sv_size; j++) {
                int offset = offset2(i, j, b.size, b.index_sv);
                phi1[offset] += dt * (phidot1[offset] + 0.5f * ker1_curr[offset] * dt);
                phi2[offset] += dt * (phidot2[offset] + 0.5f * ker2_curr[offset] * dt);
            }
        }
    }

    // tau += dt; // Do this only on the root level

    // TODO: need to update field values in the buffer
    // Since phi1/2 is technically the only field values we care about in the buffer, when we 
    // update phi1/2, we should update the relevant buffers of blocks on the same level.

    for (block_data &b : data.b_data) {
        int b_sv_size = (b.has_buffer) ? b.size - BUFFER_STENCIL : b.size;

        data_t *phi1      = &data.phi1[b.index_global];
        data_t *phi2      = &data.phi2[b.index_global];
        data_t *ker1_next = &data.ker1_next[b.index_global];
        data_t *ker2_next = &data.ker2_next[b.index_global];

        #pragma omp parallel for schedule(static) collapse(2)
        for (int i = 0; i < b_sv_size; i++) {
            for (int j = 0; j < b_sv_size; j++) {
                int offset = offset2(i, j, b.size, b.index_sv);
                data_t lap1 = laplacian2(phi1, i, j, b.size, b.index_sv) / pow_2(dx);
                data_t lap2 = laplacian2(phi2, i, j, b.size, b.index_sv) / pow_2(dx);
                ker1_next[offset] = lap1 - parameters.lambda / pow(tau_local, 2.0f * parameters.enable_PRS) * phi1[offset] * (pow_2(phi1[offset]) + pow_2(phi2[offset]) - pow_2(tau_local) + pow_2(T_initial) / (3.0f));
                ker2_next[offset] = lap2 - parameters.lambda / pow(tau_local, 2.0f * parameters.enable_PRS) * phi2[offset] * (pow_2(phi1[offset]) + pow_2(phi2[offset]) - pow_2(tau_local) + pow_2(T_initial) / (3.0f));
            }
        }
    }

    for (block_data &b : data.b_data) {
        int b_sv_size = (b.has_buffer) ? b.size - BUFFER_STENCIL : b.size;

        data_t *phidot1 = &data.phidot1[b.index_global];
        data_t *phidot2 = &data.phidot2[b.index_global];
        data_t *ker1_curr = &data.ker1_curr[b.index_global];
        data_t *ker2_curr = &data.ker2_curr[b.index_global];
        data_t *ker1_next = &data.ker1_next[b.index_global];
        data_t *ker2_next = &data.ker2_next[b.index_global];

        #pragma omp parallel for schedule(static) collapse(2)
        for (int i = 0; i < b_sv_size; i++) {
            for (int j = 0; j < b_sv_size; j++) {
                int offset = offset2(i, j, b.size, b.index_sv);
                phidot1[offset] += 0.5f * (ker1_curr[offset] + ker1_next[offset]) * dt;
                phidot2[offset] += 0.5f * (ker2_curr[offset] + ker2_next[offset]) * dt;
            }
        }

        #pragma omp parallel for schedule(static) collapse(2)
        for (int i = 0; i < b_sv_size; i++) {
            for (int j = 0; j < b_sv_size; j++) {
                int offset = offset2(i, j, b.size, b.index_sv);
                ker1_curr[offset] = ker1_next[offset];
                ker2_curr[offset] = ker2_next[offset];
            }
        }
    }

    for (block_data &b : data.b_data) {
        int b_sv_size = (b.has_buffer) ? b.size - BUFFER_STENCIL : b.size;

        data_t *phi1 = &data.phi1[b.index_global];
        data_t *phi2 = &data.phi2[b.index_global];
        int *flagged = &data.flagged[b.index_global];

        // flag points that meet refinement criterion:
        #pragma omp parallel for schedule(static) collapse(2)
        for (int i = 0; i < b_sv_size; i++) {
            for (int j = 0; j < b_sv_size; j++) {
                int offset = offset2(i, j, b.size, b.index_sv);

                // Note: don't have to divide gradient by dx^2 for refinement criterion.
                data_t grad_sq1 = gradient_squared2(phi1, i, j, b.size, b.index_sv);
                data_t grad_sq2 = gradient_squared2(phi2, i, j, b.size, b.index_sv);

                flagged[offset] = (sqrt(grad_sq1 + grad_sq2) > parameters.refinement_threshold);
            }
        }

    }

}

/**
 * Given the required block specifications, remove or add the appropriate blocks on the next level, i.e. level + 1.
 */
void regrid(std::vector<level_data*> hierarchy, int level, std::vector<std::vector<block_specs>> b_specs) { }

void reflux() {}

void average_down() {}