#include <stdio.h>
#include <assert.h>
// #include <gsl/gsl_math.h>

#include "utils/utils.h"

#include "common.h"
#include "interface.h"

static int should_save_snapshot(int tstep, int n_snapshots, int final_tstep);
static int should_count_strings(int tstep, int string_checks, int final_tstep);
static void debug(all_data data, int length, int tstep);

void run_standard() {

    all_data data;
    int length = get_length();

    open_output_filestreams();

    build_coefficient_matrix(&data.coefficient_matrix, parameters.NDIMS, parameters.N);

    set_physics_variables();

    initialise_data(&data);

    if (parameters.run_string_finding) fprintf(fp_string_finding, "time,ncores\n");

    int n_snapshots_written = 0;
    int light_time = round(0.5f * parameters.N * parameters.space_step / parameters.time_step);
    int final_step = light_time - round(1 * parameters.space_step / parameters.time_step) + 1;

    for (int tstep = 0; tstep < final_step; tstep++) {

        debug(data, length, tstep);

        velocity_verlet_scheme(data);

        float kappa = string_tension(tau);
        float time = physical_time(tau);
        float Hinv = 1.0f / hubble_parameter(tau);

        if (should_count_strings(tstep, parameters.string_checks, final_step)) {

            // compute axion field: 
            for (int i = 0; i < length; i++) data.axion[i] = atan2(data.phi1[i], data.phi2[i]);

            if (parameters.NDIMS == 2) {
                std::vector <vec2i> s;
                int num_cores = Cores2D(data.axion, &s);
                float xi = num_cores * pow_2(time / tau);
                fprintf(fp_string_finding, "%f, %d\n",time,num_cores);
            }

            if (parameters.NDIMS == 3) {
                std::vector <vec3i> s;
                int num_cores = Cores3D(data.axion, &s);
                float xi = num_cores * pow_2(time / tau);
                fprintf(fp_string_finding, "%f, %d\n",time,num_cores);
            }
        }

        if (should_save_snapshot(tstep, parameters.n_snapshots, final_step)) {
            fprintf(fp_main_output, "Writing snapshot %d:\n", n_snapshots_written);

            // output time variables:
            fprintf(fp_main_output, "  string tension = %f\n", kappa);
            fprintf(fp_main_output, "  time           = %f\n", time);

            // file names:
            char fname_phi1[50], fname_phi2[50], fname_strings[50];
            sprintf(fname_phi1, "phi1-snapshot%d", n_snapshots_written);
            sprintf(fname_phi2, "phi2-snapshot%d", n_snapshots_written);
            sprintf(fname_strings, "string-pos-snapshot%d", n_snapshots_written);

            save_data(fname_phi1, data.phi1, length);
            save_data(fname_phi2, data.phi2, length);

            // compute axion field: 
            for (int i = 0; i < length; i++) data.axion[i] = atan2(data.phi1[i], data.phi2[i]);

            // save string positions:
            if (parameters.NDIMS == 2) {
                std::vector <vec2i> s;
                Cores2D(data.axion, &s);
                save_strings2(fname_strings, &s);
            }
            if (parameters.NDIMS == 3) {
                std::vector <vec3i> s;
                Cores3D(data.axion, &s);
                save_strings3(fname_strings, &s);
            }

            n_snapshots_written++;
        }
    }

    // Free memory:
    free_all_data(data);

    close_output_filestreams();
}


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

static int should_count_strings(int tstep, int string_checks, int final_tstep) {
    if (!parameters.run_string_finding) return 0;
    // TODO: temporary solution
    return tstep == 0 || tstep % (final_tstep / (string_checks - 1)) == 0;
}
