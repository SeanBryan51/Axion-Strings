#include "amr_internal.hpp"
#include "amr_interface.hpp"

static void debug(level_data data, int length, int tstep) {
    for (int i = 0; i < length; i++) {
        if (isnan(data.phi1[i]) || isnan(data.phi2[i])) {
            fprintf(fp_main_output, "Error: NaN encountered in solution vector.\n");
            fprintf(fp_main_output, " tstep = %d\n", tstep);
            assert(0);
        }
    }
}

void regrid(std::vector<level_data> hierarchy) {
    return;
}

void reflux() {
    return;
}

void average_down() {
    return;
}

/*
 * Implementation of the Berger-Collela time-stepping algorithm:
 * Function takes in a grid hierarchy and evolves the level
 * specified by level by one time step.
 */
void evolve_level(std::vector<level_data> hierarchy, int level) {

    integrate_level(hierarchy, level); // integrate current level without knowing the fine level data

    if (1) {
        // Should refine grid:

        // Called at a specific timestep (not everytime as it is computationally expensive?).
        // Solution: integrate_level() should check the refinement condition after integrating fields.
        
        // Buffers allow features to move within a patch between refinement periods. Can also gamble by using
        // a very accurate refinement criterion so we can be sure we don't lose resolution of features in between 
        // refinement periods?
        // Solution: for now just check after every integration.
        regrid(hierarchy);
    }

    if (level + 1 < hierarchy.size()) {
        // There exists a finer level:

        // Perform subcycling:
        // For a factor of 2 refinement between levels, we need to evolve the level twice:
        evolve_level(hierarchy, level + 1);
        evolve_level(hierarchy, level + 1);

        reflux();
        average_down(); // set covered coarse cells to be the average of fine
    }

}

void run_amr() {

    std::vector<level_data> hierarchy;
    level_data root_level;
    root_level.size = get_length();
    root_level.b_index = {0}; // the root grid starts at index 0
    root_level.b_size = {parameters.N};

    set_physics_variables();

    // Sanity check on input parameters:
    assert(parameters.N % 2 == 0); // Number of grid points should always be some power of 2.
    assert(parameters.NDIMS == 2 || parameters.NDIMS == 3);

    root_level.phi1      = (data_t *) calloc(root_level.size, sizeof(data_t));
    root_level.phi2      = (data_t *) calloc(root_level.size, sizeof(data_t));
    root_level.phidot1   = (data_t *) calloc(root_level.size, sizeof(data_t));
    root_level.phidot2   = (data_t *) calloc(root_level.size, sizeof(data_t));
    root_level.ker1_curr = (data_t *) calloc(root_level.size, sizeof(data_t));
    root_level.ker2_curr = (data_t *) calloc(root_level.size, sizeof(data_t));
    root_level.ker1_next = (data_t *) calloc(root_level.size, sizeof(data_t));
    root_level.ker2_next = (data_t *) calloc(root_level.size, sizeof(data_t));
    root_level.axion = NULL;
    root_level.saxion = NULL;

    assert(root_level.phi1 != NULL && root_level.phi2 != NULL && root_level.phidot1 != NULL && root_level.phidot2 != NULL);
    assert(root_level.ker1_curr != NULL && root_level.ker2_curr != NULL && root_level.ker1_next != NULL && root_level.ker2_next != NULL);

    // Set initial field values:
    gaussian_thermal(root_level.phi1, root_level.phi2, root_level.phidot1, root_level.phidot2);

    hierarchy = {root_level};

    int final_step = round(light_crossing_time / parameters.time_step) - round(parameters.space_step / parameters.time_step) + 1;
    for (int tstep = 0; tstep < final_step; tstep++) {
        
        evolve_level(hierarchy, 0);

        debug(root_level, root_level.size, tstep);
    }

    // Clean up memory:
    for (level_data level : hierarchy) {
        free(level.phi1);
        free(level.phi2);
        free(level.phidot1);
        free(level.phidot2);
        free(level.ker1_curr);
        free(level.ker2_curr);
        free(level.ker1_next);
        free(level.ker2_next);
        free(level.axion);
        free(level.saxion);
    }
}
