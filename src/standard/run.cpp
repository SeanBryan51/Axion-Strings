#include <stdio.h>
#include <assert.h>
#include <gsl/gsl_math.h>

#include "utils/utils.h"

#include "common.h"
#include "interface.h"

int should_save_snapshot(int tstep, int n_snapshots, int final_tstep);
int should_count_strings(int tstep, int string_checks, int final_tstep);
void debug(all_data data, int length, int tstep);
void run_standard() {

    all_data data;
    int length = get_length();

    initialise_everything(&data);

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
                int num_cores = Cores2D(data.axion, parameters.thr);
                float xi = num_cores * gsl_pow_2(time / tau);
                // printf("%f %f\n",time,xi);
                printf("%f %f\n",Hinv,xi);
            }
            if (parameters.NDIMS == 3) {
                int num_cores = Cores3D(data.axion, parameters.thr);
                float xi = num_cores * gsl_pow_2(time / tau);
                printf("%f %f\n",time,xi);
            }
        }

        if (should_save_snapshot(tstep, parameters.n_snapshots, final_step)) {
            printf("Writing snapshot %d:\n", n_snapshots_written);

            // output time variables:
            printf("  string tension = %f\n", kappa);
            printf("  time           = %f\n", time);

            // file names:
            char fname_phi1[50], fname_phi2[50];
            sprintf(fname_phi1, "phi1-snapshot%d", n_snapshots_written);
            sprintf(fname_phi2, "phi2-snapshot%d", n_snapshots_written);

            save_data(fname_phi1, data.phi1, length);
            save_data(fname_phi2, data.phi2, length);

            n_snapshots_written++;
        }
    }

    // Free memory:
    free_all_data(data);
}


void debug(all_data data, int length, int tstep) {
    for (int i = 0; i < length; i++) {
        if (gsl_isnan(data.phi1[i]) || gsl_isnan(data.phi2[i])) {
            printf("Error: NaN encountered in solution vector.\n");
            printf(" tstep = %d\n", tstep);
            assert(0);
        }
    }
}

int should_save_snapshot(int tstep, int n_snapshots, int final_tstep) {
    if (!parameters.save_snapshots) return 0;
    // TODO: temporary solution
    return tstep == 0 || tstep % (final_tstep / (n_snapshots - 1)) == 0;
}

int should_count_strings(int tstep, int string_checks, int final_tstep) {
    if (!parameters.run_string_finding) return 0;
    // TODO: temporary solution
    return tstep == 0 || tstep % (final_tstep / (string_checks - 1)) == 0;
}
