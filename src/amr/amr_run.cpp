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

    root_level.flagged = (int *) calloc(root_level.size, sizeof(int));

    assert(root_level.flagged != NULL);

    // Set initial field values:
    gaussian_thermal(root_level.phi1, root_level.phi2, root_level.phidot1, root_level.phidot2);

    hierarchy = {root_level};

    int final_step = round(light_crossing_time / parameters.time_step) - round(parameters.space_step / parameters.time_step) + 1;
    for (int tstep = 0; tstep < final_step; tstep++) {

        evolve_level(hierarchy, 0, tau);

        tau += parameters.time_step;

        debug(root_level, root_level.size, tstep);
    }

#if 0
    std::vector<vec2i> block_coords;
    std::vector<int> block_size;

    gen_refinement_blocks(block_coords, block_size, hierarchy, 0);

    assert(block_coords.size() == block_size.size());
    for (int i = 0; i < block_coords.size(); i++) {
        printf("(%d, %d),\n", block_coords[i].x, block_coords[i].y);
        printf("(%d, %d),\n", block_coords[i].x + block_size[i] - 1, block_coords[i].y + block_size[i] - 1);
        // printf("size = %d\n", block_size[i]);
    }
#endif

#if 0
    FILE *fp = fopen("/Users/seanbryan/Documents/UNI/2021T1-2/Project/Axion-Strings/output_files/snapshot-flagged", "w");
    assert(fp != NULL);
    fwrite(root_level.flagged, sizeof(int), root_level.size, fp);
    fclose(fp);
#endif

#if 0
    save_data("snapshot-test-phi1", root_level.phi1, root_level.size);
    save_data("snapshot-test-phi2", root_level.phi2, root_level.size);
#endif

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
