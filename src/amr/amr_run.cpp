#include "amr_internal.hpp"
#include "amr_interface.hpp"

const char *phi1_ic_path    = "/Users/seanbryan/Documents/UNI/2021T1-2/Project/Axion-Strings/output_files/4-strings-ic-2DN128-TAU64/snapshot-final-phi1";
const char *phi2_ic_path    = "/Users/seanbryan/Documents/UNI/2021T1-2/Project/Axion-Strings/output_files/4-strings-ic-2DN128-TAU64/snapshot-final-phi2";
const char *phidot1_ic_path = "/Users/seanbryan/Documents/UNI/2021T1-2/Project/Axion-Strings/output_files/4-strings-ic-2DN128-TAU64/snapshot-final-phidot1";
const char *phidot2_ic_path = "/Users/seanbryan/Documents/UNI/2021T1-2/Project/Axion-Strings/output_files/4-strings-ic-2DN128-TAU64/snapshot-final-phidot2";

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

static level_data *create_test_level(level_data *root_level) {

    level_data *ret = (level_data *) calloc(1, sizeof(level_data));
    ret->length = (parameters.N + BUFFER_STENCIL) * (parameters.N + BUFFER_STENCIL);
    ret->tau_int = 0;
    ret->b_data = { (block_data) {
        .index_global = 0,
        .index_sv = (parameters.N + BUFFER_STENCIL + 1) * BUFFER_STENCIL / 2,
        .size = parameters.N + BUFFER_STENCIL,
        .has_buffer = 1,
        .origin_global = {0, 0} // in units of level 1
    } };

    ret->phi1      = (data_t *) calloc(ret->length, sizeof(data_t));
    ret->phi2      = (data_t *) calloc(ret->length, sizeof(data_t));
    ret->phidot1   = (data_t *) calloc(ret->length, sizeof(data_t));
    ret->phidot2   = (data_t *) calloc(ret->length, sizeof(data_t));
    ret->ker1_curr = (data_t *) calloc(ret->length, sizeof(data_t));
    ret->ker2_curr = (data_t *) calloc(ret->length, sizeof(data_t));
    ret->ker1_next = (data_t *) calloc(ret->length, sizeof(data_t));
    ret->ker2_next = (data_t *) calloc(ret->length, sizeof(data_t));
    ret->axion = NULL;
    ret->saxion = NULL;

    assert(ret->phi1 != NULL && ret->phi2 != NULL && ret->phidot1 != NULL && ret->phidot2 != NULL);
    assert(ret->ker1_curr != NULL && ret->ker2_curr != NULL && ret->ker1_next != NULL && ret->ker2_next != NULL);

    ret->flagged = (int *) calloc(ret->length, sizeof(int));

    assert(ret->flagged != NULL);

    ret->phi1_prev = (data_t *) calloc(ret->length, sizeof(data_t));
    ret->phi2_prev = (data_t *) calloc(ret->length, sizeof(data_t));

    assert(ret->phi1_prev != NULL && ret->phi2_prev != NULL);

    // Fill solution vectors by interpolating from coarse grid:
    block_data b = ret->b_data[0];
    int b_sv_size = (b.has_buffer) ? b.size - BUFFER_STENCIL : b.size;
    for (int i = 0; i < b_sv_size; i++) {
        for (int j = 0; j < b_sv_size; j++) {
            vec2i global_coordinate = coordinate_global(i, j, b);

            global_to_local_return_t lc = coordinate_global_to_local(1, global_coordinate, 0, root_level->b_data[0]);
            int i_root_level = lc.local_coordinate.x;
            int j_root_level = lc.local_coordinate.y;
            int x_offset = lc.offsets.x;
            int y_offset = lc.offsets.y;

            data_t conversion_factor = pow(REFINEMENT_FACTOR, 1 - 0);

            ret->phi1[offset2(i, j, b.size, b.index_global + b.index_sv)] = interpolate2(root_level->phi1, i_root_level, j_root_level, x_offset, y_offset, root_level->b_data[0].size, root_level->b_data[0].index_global + root_level->b_data[0].index_sv, conversion_factor);

            ret->phi2[offset2(i, j, b.size, b.index_global + b.index_sv)] = interpolate2(root_level->phi2, i_root_level, j_root_level, x_offset, y_offset, root_level->b_data[0].size, root_level->b_data[0].index_global + root_level->b_data[0].index_sv, conversion_factor);

            ret->phidot1[offset2(i, j, b.size, b.index_global + b.index_sv)] = interpolate2(root_level->phidot1, i_root_level, j_root_level, x_offset, y_offset, root_level->b_data[0].size, root_level->b_data[0].index_global + root_level->b_data[0].index_sv, conversion_factor);

            ret->phidot2[offset2(i, j, b.size, b.index_global + b.index_sv)] = interpolate2(root_level->phidot2, i_root_level, j_root_level, x_offset, y_offset, root_level->b_data[0].size, root_level->b_data[0].index_global + root_level->b_data[0].index_sv, conversion_factor);
        }
    }

    return ret;
}

void run_amr() {

    std::vector<level_data *> hierarchy;
    level_data *root_level = (level_data *) calloc(1, sizeof(level_data));
    root_level->length = get_length();
    root_level->tau_int = 0;
    root_level->b_data = { (block_data) {
        .index_global = 0,
        .index_sv = 0,
        .size = parameters.N,
        .has_buffer = 0,
        .origin_global = {0, 0}
    } };

    hierarchy = { root_level };

    set_physics_variables();

    // Sanity check on input parameters:
    assert(parameters.N % 2 == 0); // Number of grid points should always be some power of 2.
    assert(parameters.NDIMS == 2);

    root_level->phi1      = (data_t *) calloc(root_level->length, sizeof(data_t));
    root_level->phi2      = (data_t *) calloc(root_level->length, sizeof(data_t));
    root_level->phidot1   = (data_t *) calloc(root_level->length, sizeof(data_t));
    root_level->phidot2   = (data_t *) calloc(root_level->length, sizeof(data_t));
    root_level->ker1_curr = (data_t *) calloc(root_level->length, sizeof(data_t));
    root_level->ker2_curr = (data_t *) calloc(root_level->length, sizeof(data_t));
    root_level->ker1_next = (data_t *) calloc(root_level->length, sizeof(data_t));
    root_level->ker2_next = (data_t *) calloc(root_level->length, sizeof(data_t));
    root_level->axion = NULL;
    root_level->saxion = NULL;

    assert(root_level->phi1 != NULL && root_level->phi2 != NULL && root_level->phidot1 != NULL && root_level->phidot2 != NULL);
    assert(root_level->ker1_curr != NULL && root_level->ker2_curr != NULL && root_level->ker1_next != NULL && root_level->ker2_next != NULL);

    root_level->flagged = (int *) calloc(root_level->length, sizeof(int));

    assert(root_level->flagged != NULL);

    root_level->phi1_prev = (data_t *) calloc(root_level->length, sizeof(data_t));
    root_level->phi2_prev = (data_t *) calloc(root_level->length, sizeof(data_t));

    assert(root_level->phi1_prev != NULL && root_level->phi2_prev != NULL);

    // Set initial field values:
    if (parameters.init_from_snapshot) {
        fio_read_field_data(phi1_ic_path, root_level->phi1, root_level->length);
        fio_read_field_data(phi2_ic_path, root_level->phi2, root_level->length);
        fio_read_field_data(phidot1_ic_path, root_level->phidot1, root_level->length);
        fio_read_field_data(phidot2_ic_path, root_level->phidot2, root_level->length);
        tau = parameters.tau_initial;
        if (!parameters.enable_PRS) parameters.lambda /= pow_2(tau); // Need to adjust value of quartic coupling for physical simulation.
    } else {
        gaussian_thermal(root_level->phi1, root_level->phi2, root_level->phidot1, root_level->phidot2);
    }

    level_data *new_level = create_test_level(root_level);
    hierarchy.push_back(new_level);

    if (parameters.save_snapshots) {
        fprintf(fp_snapshot_timings, "snapshot,");
        fprintf(fp_snapshot_timings, "tau,");
        fprintf(fp_snapshot_timings, "hubble_scale,");
        fprintf(fp_snapshot_timings, "string_tension,");
        fprintf(fp_snapshot_timings, "\n");
    }

    int n_snapshots_written = 0;

    int final_step = round(light_crossing_time / parameters.time_step) - round(parameters.space_step / parameters.time_step) + 1;
    // final_step *= 2; // to observe shrinking of string cores
    for (int tstep = 0; tstep < final_step; tstep++) {

        if (should_save_snapshot(tstep, parameters.n_snapshots, final_step)) {
            fprintf(fp_main_output, "Writing snapshot %d:\n", n_snapshots_written);

            // output snapshot timings:
            fprintf(fp_snapshot_timings, "%d,", n_snapshots_written);
            fprintf(fp_snapshot_timings, "%f,", tau);
            fprintf(fp_snapshot_timings, "%f,", 1.0f / hubble_parameter());
            fprintf(fp_snapshot_timings, "%f,", (parameters.lambda != 0.0f) ? string_tension() : 0.0f);
            fprintf(fp_snapshot_timings, "\n");

            printf("tau_int = %d\n", root_level->tau_int);
            if (parameters.save_fields) {

                char fname_phi1[50], fname_phi2[50];
                sprintf(fname_phi1, "snapshot%d-phi1", n_snapshots_written);
                sprintf(fname_phi2, "snapshot%d-phi2", n_snapshots_written);

                // fio_save_field_data(fname_phi1, root_level->phi1, root_level->length);
                // fio_save_field_data(fname_phi2, root_level->phi2, root_level->length);
                fio_save_field_data(fname_phi1, new_level->phi1, new_level->length);
                fio_save_field_data(fname_phi2, new_level->phi2, new_level->length);
            }

#if 0
            // debug
            // output flagged points:
            char fname_flagged[50];
            sprintf(fname_flagged, "snapshot%d-flagged", n_snapshots_written);
            fio_save_flagged_data(fname_flagged, root_level.flagged, root_level.length);
#endif

            n_snapshots_written++;
        }

#if 0
        // debug
        if (n_snapshots_written > 4) break;
#endif

        evolve_level(hierarchy, 0);
        root_level->tau_int += 1;

        tau += parameters.time_step;

        debug(*root_level, root_level->length, tstep);
    }

#if 0
    // debug
    FILE *fp = fopen("/Users/seanbryan/Documents/UNI/2021T1-2/Project/Axion-Strings/output_files/snapshot-flagged", "w");
    assert(fp != NULL);
    fwrite(root_level.flagged, sizeof(int), root_level.length, fp);
    fclose(fp);
#endif

#if 0
    assert(root_level.b_data.size() == 1);

    std::vector<block_spec_t> to_refine = {};

    gen_refinement_blocks(to_refine, root_level.flagged, root_level.b_data[0]);

    for (int i = 0; i < to_refine.size(); i++) {
        printf("[(%d, %d),", to_refine[i].coord.x, to_refine[i].coord.y);
        printf("(%d, %d)],\n", to_refine[i].coord.x + (to_refine[i].size - 1), to_refine[i].coord.y + (to_refine[i].size - 1));
    }
#endif

    // Clean up memory:
    for (level_data *data : hierarchy) {
        free(data->phi1);
        free(data->phi2);
        free(data->phidot1);
        free(data->phidot2);
        free(data->ker1_curr);
        free(data->ker2_curr);
        free(data->ker1_next);
        free(data->ker2_next);
        free(data->axion);
        free(data->saxion);
        free(data->flagged);
        free(data);
    }
}
