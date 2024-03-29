#include "s_internal.hpp"
#include "s_interface.hpp"

static void debug(all_data data, int length, int tstep) {
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

static int should_sample_time_series(int tstep, int n_samples, int final_tstep) {
    if (!parameters.sample_time_series) return 0;
    // TODO: temporary solution
    return tstep == 0 || tstep % (final_tstep / (n_samples - 1)) == 0;
}

void run_standard() {

    all_data data;
    int length = get_length();

    // Sanity check on input parameters:
    assert(parameters.N % 2 == 0); // Number of grid points should always be some power of 2.
    assert(parameters.NDIMS == 2 || parameters.NDIMS == 3);

    build_coefficient_matrix(&data.coefficient_matrix, parameters.NDIMS, parameters.N);

    set_physics_variables();

    // Allocate fields on the heap:
    data.phi1      = (data_t *) calloc(length, sizeof(data_t));
    data.phi2      = (data_t *) calloc(length, sizeof(data_t));
    data.phidot1   = (data_t *) calloc(length, sizeof(data_t));
    data.phidot2   = (data_t *) calloc(length, sizeof(data_t));
    data.ker1_curr = (data_t *) calloc(length, sizeof(data_t));
    data.ker2_curr = (data_t *) calloc(length, sizeof(data_t));
    data.ker1_next = (data_t *) calloc(length, sizeof(data_t));
    data.ker2_next = (data_t *) calloc(length, sizeof(data_t));
    data.axion = NULL;
    data.saxion = NULL;

    // Assert allocations were successful:
    assert(data.phi1 != NULL && data.phi2 != NULL && data.phidot1 != NULL && data.phidot2 != NULL);
    assert(data.ker1_curr != NULL && data.ker2_curr != NULL && data.ker1_next != NULL && data.ker2_next != NULL);

    if ((parameters.save_snapshots && parameters.save_strings) || (parameters.sample_time_series && parameters.sample_strings)) {
        data.axion = (data_t *) calloc(length, sizeof(data_t));
        assert(data.axion != NULL);
    }

    // Set initial field values:
    gaussian_thermal(data.phi1, data.phi2, data.phidot1, data.phidot2);

    if (parameters.save_snapshots) {
        fprintf(fp_snapshot_timings, "snapshot,");
        fprintf(fp_snapshot_timings, "tau,");
        fprintf(fp_snapshot_timings, "hubble_scale,");
        fprintf(fp_snapshot_timings, "string_tension,");
        fprintf(fp_snapshot_timings, "\n");
    }

    if (parameters.sample_time_series) {
        fprintf(fp_time_series, "time,");
        fprintf(fp_time_series, "scalefactor,");
        if (parameters.sample_strings) fprintf(fp_time_series, "xi,");
        if (parameters.sample_strings) fprintf(fp_time_series, "n_plaquettes,");
        if (parameters.sample_background) fprintf(fp_time_series, "phi1_bar,phi2_bar,phidot1_bar,phidot2_bar,axion_bar,saxion_bar");
        fprintf(fp_time_series, "\n");
    }

    int n_snapshots_written = 0;
    // Note: should use the light crossing time in conformal time instead of in physical time however this still works well.
    int final_step = round(light_crossing_time / parameters.time_step) - round(parameters.space_step / parameters.time_step) + 1;
    for (int tstep = 0; tstep < final_step; tstep++) {

        if (should_sample_time_series(tstep, parameters.n_samples, final_step)) {

            fprintf(fp_time_series, "%f", physical_time());
            fprintf(fp_time_series, ",%f", scale_factor());

            if (parameters.sample_strings) {

                // compute axion field:
                #pragma omp parallel for schedule(static)
                for (int i = 0; i < length; i++) data.axion[i] = atan2(data.phi1[i], data.phi2[i]);

                if (parameters.NDIMS == 2) {
                    // 2-dimensions:
                    std::vector <vec2i> s;
                    int n_plaquettes = cores2(data.axion, s);

                    // Count neighbouring plaquettes:
                    int count = 0;
                    for (int i = 0; i < n_plaquettes - 1; i++) {
                        int diff_y = s.at(i+1).y - s.at(i).y;
                        int diff_x = s.at(i+1).x - s.at(i).x;
                        if (diff_y == 0 && diff_x == 1) count += 1;
                        if (diff_y == 1 && diff_x == 0) count += 1;
                    }
                    int num_cores = n_plaquettes - count;

                    float t_phys = physical_time();
                    float a = scale_factor();
                    float physical_area = pow_2(parameters.N * parameters.space_step * a);
                    float xi = num_cores * pow_2(t_phys) / physical_area;
                    fprintf(fp_time_series, ",%f", xi);
                    fprintf(fp_time_series, ",%d", n_plaquettes);

                } else {
                    // 3-dimensions:
                    std::vector <vec3i> s;
                    int n_plaquettes = cores3(data.axion, s);

                    float total_length = n_plaquettes * 2.0f / 3.0f; // See Moore paper (arXiv:1509.00026) for factor 2/3 statistical correction.

                    float t_phys = physical_time();
                    float a = scale_factor();
                    float physical_length = total_length * a;
                    float physical_volume = pow_3(parameters.N * parameters.space_step * a);
                    float xi = physical_length * pow_2(t_phys) / physical_volume;
                    fprintf(fp_time_series, ",%f", xi);
                    fprintf(fp_time_series, ",%d", n_plaquettes);
                }
            }

            if (parameters.sample_background) {
                data_t phi1_bar, phi2_bar, phidot1_bar, phidot2_bar, axion_bar, saxion_bar;
                phi1_bar = phi2_bar = phidot1_bar = phidot2_bar = axion_bar = saxion_bar = 0.0f;

                #pragma omp parallel for schedule(static) reduction(+:phi1_bar,phi2_bar,phidot1_bar,phidot2_bar,axion_bar,saxion_bar)
                for (int i = 0; i < length; i++) {
                    phi1_bar += data.phi1[i];
                    phi2_bar += data.phi2[i];
                    phidot1_bar += data.phidot1[i];
                    phidot2_bar += data.phidot2[i];
                    axion_bar += abs(atan2(data.phi1[i], data.phi2[i]));
                    saxion_bar += sqrt(pow_2(data.phi1[i]) + pow_2(data.phi2[i]));
                }

                fprintf(fp_time_series, ",%f,%f,%f,%f,%f,%f", phi1_bar / length, phi2_bar / length, phidot1_bar / length, phidot2_bar / length, axion_bar / length, saxion_bar / length);
            }

            fprintf(fp_time_series, "\n");
        }

        if (should_save_snapshot(tstep, parameters.n_snapshots, final_step)) {

            fprintf(fp_main_output, "Writing snapshot %d:\n", n_snapshots_written);

            // output snapshot timings:
            fprintf(fp_snapshot_timings, "%d,", n_snapshots_written);
            fprintf(fp_snapshot_timings, "%f,", tau);
            fprintf(fp_snapshot_timings, "%f,", 1.0f / hubble_parameter());
            fprintf(fp_snapshot_timings, "%f,", (parameters.lambda != 0.0f) ? string_tension() : 0.0f);
            fprintf(fp_snapshot_timings, "\n");

            if (parameters.save_fields) {

                char fname_phi1[50], fname_phi2[50];
                sprintf(fname_phi1, "snapshot%d-phi1", n_snapshots_written);
                sprintf(fname_phi2, "snapshot%d-phi2", n_snapshots_written);

                if (parameters.NDIMS == 2) {
                    fio_save_field_data(fname_phi1, data.phi1, length);
                    fio_save_field_data(fname_phi2, data.phi2, length);
                }

                if (parameters.NDIMS == 3) {
                    fio_save_field_data_as_slice(fname_phi1, data.phi1, length, parameters.N);
                    fio_save_field_data_as_slice(fname_phi2, data.phi2, length, parameters.N);
                }
            }

            if (parameters.save_strings) {

                char fname_strings[50];
                sprintf(fname_strings, "snapshot%d-strings.csv", n_snapshots_written);

                #pragma omp parallel for schedule(static)
                for (int i = 0; i < length; i++) data.axion[i] = atan2(data.phi1[i], data.phi2[i]);

                if (parameters.NDIMS == 2) {
                    std::vector <vec2i> s;
                    cores2(data.axion, s);
                    fio_save_strings2(fname_strings, &s);
                } else {
                    std::vector <vec3i> s;
                    cores3(data.axion, s);
                    fio_save_strings3(fname_strings, &s);
                }
            }

            if (parameters.save_pk) {

                char fname_pk[50];
                sprintf(fname_pk, "snapshot%d-pk.csv", n_snapshots_written);
                output_powerspec(fname_pk, data.phi2, NULL);
            }


            n_snapshots_written++;
        }

        // debug(data, length, tstep);

        vvsl_field_rescaled(data);
        // vvsl_hamiltonian_form(data);

    }

    // Free memory:
    free(data.phi1);
    free(data.phi2);
    free(data.phidot1);
    free(data.phidot2);
    free(data.ker1_curr);
    free(data.ker2_curr);
    free(data.ker1_next);
    free(data.ker2_next);
    free(data.axion);
    free(data.saxion);
    mkl_sparse_destroy(data.coefficient_matrix);
}
