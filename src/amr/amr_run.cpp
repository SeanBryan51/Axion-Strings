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

static int should_save_snapshot(int tstep, int n_snapshots, int final_tstep) {
    if (!parameters.save_snapshots) return 0;
    // TODO: temporary solution
    return tstep == 0 || tstep % (final_tstep / (n_snapshots - 1)) == 0;
}

void run_amr() {

    std::vector<level_data> hierarchy;
    level_data root_level;
    root_level.length = get_length();
    root_level.b_data = { (block_data) {.index = 0, .buffer_index = 0, .size = parameters.N, .has_buffer = 0} };

    set_physics_variables();

    // Sanity check on input parameters:
    assert(parameters.N % 2 == 0); // Number of grid points should always be some power of 2.
    assert(parameters.NDIMS == 2 || parameters.NDIMS == 3);

    root_level.phi1      = (data_t *) calloc(root_level.length, sizeof(data_t));
    root_level.phi2      = (data_t *) calloc(root_level.length, sizeof(data_t));
    root_level.phidot1   = (data_t *) calloc(root_level.length, sizeof(data_t));
    root_level.phidot2   = (data_t *) calloc(root_level.length, sizeof(data_t));
    root_level.ker1_curr = (data_t *) calloc(root_level.length, sizeof(data_t));
    root_level.ker2_curr = (data_t *) calloc(root_level.length, sizeof(data_t));
    root_level.ker1_next = (data_t *) calloc(root_level.length, sizeof(data_t));
    root_level.ker2_next = (data_t *) calloc(root_level.length, sizeof(data_t));
    root_level.axion = NULL;
    root_level.saxion = NULL;

    assert(root_level.phi1 != NULL && root_level.phi2 != NULL && root_level.phidot1 != NULL && root_level.phidot2 != NULL);
    assert(root_level.ker1_curr != NULL && root_level.ker2_curr != NULL && root_level.ker1_next != NULL && root_level.ker2_next != NULL);

    root_level.flagged = (int *) calloc(root_level.length, sizeof(int));

    assert(root_level.flagged != NULL);

    // Set initial field values:
    if (parameters.init_from_snapshot) {
        fio_read_field_data("4-strings-ic-2DN128-TAU64/snapshot-final-phi1", root_level.phi1, root_level.length);
        fio_read_field_data("4-strings-ic-2DN128-TAU64/snapshot-final-phi2", root_level.phi2, root_level.length);
        fio_read_field_data("4-strings-ic-2DN128-TAU64/snapshot-final-phidot1", root_level.phidot1, root_level.length);
        fio_read_field_data("4-strings-ic-2DN128-TAU64/snapshot-final-phidot2", root_level.phidot2, root_level.length);
        tau = parameters.tau_initial;
        parameters.lambdaPRS /= pow_2(tau);
    } else {
        gaussian_thermal(root_level.phi1, root_level.phi2, root_level.phidot1, root_level.phidot2);
    }

    hierarchy = {root_level};

    if (parameters.save_snapshots) {
        fprintf(fp_snapshot_timings, "snapshot,");
        fprintf(fp_snapshot_timings, "tau,");
        fprintf(fp_snapshot_timings, "hubble_scale,");
        fprintf(fp_snapshot_timings, "string_tension,");
        fprintf(fp_snapshot_timings, "\n");
    }

    int n_snapshots_written = 0;

    int final_step = round(light_crossing_time / parameters.time_step) - round(parameters.space_step / parameters.time_step) + 1;
    final_step *= 2; // (to observe shrinking of string cores)
    for (int tstep = 0; tstep < final_step; tstep++) {

        if (should_save_snapshot(tstep, parameters.n_snapshots, final_step)) {
            fprintf(fp_main_output, "Writing snapshot %d:\n", n_snapshots_written);

            // output snapshot timings:
            fprintf(fp_snapshot_timings, "%d,", n_snapshots_written);
            fprintf(fp_snapshot_timings, "%f,", tau);
            fprintf(fp_snapshot_timings, "%f,", 1.0f / hubble_parameter());
            fprintf(fp_snapshot_timings, "%f,", (parameters.lambdaPRS != 0.0f) ? string_tension() : 0.0f);
            fprintf(fp_snapshot_timings, "\n");

            if (parameters.save_fields) {

                char fname_phi1[50], fname_phi2[50];
                sprintf(fname_phi1, "snapshot%d-phi1", n_snapshots_written);
                sprintf(fname_phi2, "snapshot%d-phi2", n_snapshots_written);

                fio_save_field_data(fname_phi1, root_level.phi1, root_level.length);
                fio_save_field_data(fname_phi2, root_level.phi2, root_level.length);
            }
#if 0
            // output flagged points:
            char fname_flagged[50];
            sprintf(fname_flagged, "snapshot%d-flagged", n_snapshots_written);
            fio_save_flagged_data(fname_flagged, root_level.flagged, root_level.length);
#endif
            n_snapshots_written++;
        }

        evolve_level(hierarchy, 0, tau);

        tau += parameters.time_step;

        debug(root_level, root_level.length, tstep);
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
    fio_save_field_data("snapshot-test-phi1", root_level.phi1, root_level.size);
    fio_save_field_data("snapshot-test-phi2", root_level.phi2, root_level.size);
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
