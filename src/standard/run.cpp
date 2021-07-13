#include <stdio.h>
#include <assert.h>
#include <gsl/gsl_math.h>

#include "../parameters.h"
#include "../utils/utils.h"

#include "common.h"
#include "interface.h"

int should_save_snapshot(int tstep, int n_snapshots, int final_tstep);
int should_count_strings(int tstep, int string_checks, int final_tstep);
void debug(dtype *phi1, dtype *phi2,
           dtype *phidot1, dtype *phidot2,
           dtype *ker1_curr, dtype *ker2_curr,
           dtype *ker1_next, dtype *ker2_next,
           int length, int tstep);

void run_standard() {

    set_internal_variables();

    if (parameters.use_coeff_matrix) set_coefficient_matrix(parameters.coeff_matrix_path, &coefficient_matrix);
    else build_coefficient_matrix(&coefficient_matrix, parameters.NDIMS, parameters.N);

    // Allocate fields on the heap:

    assert(parameters.NDIMS == 2 || parameters.NDIMS == 3);
    int N = parameters.N;
    int length = (parameters.NDIMS == 3) ? (N * N * N) : (N * N);

    dtype *phi1 = (dtype *) calloc(length, sizeof(dtype));
    dtype *phi2 = (dtype *) calloc(length, sizeof(dtype));
    dtype *phidot1 = (dtype *) calloc(length, sizeof(dtype));
    dtype *phidot2 = (dtype *) calloc(length, sizeof(dtype));

    dtype *axion = (dtype *) calloc(length, sizeof(dtype));

    // Assert allocation was successful:
    assert(phi1 != NULL && phi2 != NULL && phidot1 != NULL && phidot2 != NULL);

    // Initialise fields:
    // init_noise(phi1, phi2, phidot1, phidot2);
    gaussian_thermal(phi1, phi2, phidot1, phidot2);

    // Allocate field kernels on the heap:
    dtype *ker1_curr = (dtype *) calloc(length, sizeof(dtype));
    dtype *ker2_curr = (dtype *) calloc(length, sizeof(dtype));
    dtype *ker1_next = (dtype *) calloc(length, sizeof(dtype));
    dtype *ker2_next = (dtype *) calloc(length, sizeof(dtype));

    // Assert allocation was successful:
    assert(ker1_curr != NULL && ker2_curr != NULL && ker1_next != NULL && ker2_next != NULL);

    // Initialise kernels:
    kernels(ker1_next, ker2_next, phi1, phi2, phidot1, phidot2);

    int n_snapshots_written = 0;
    int light_time = round(0.5f * parameters.N * parameters.space_step / parameters.time_step);
    int final_step = light_time - round(1 * parameters.space_step / parameters.time_step) + 1;
    for (int tstep = 0; tstep < final_step; tstep++) {

        debug(phi1, phi2, phidot1, phidot2, ker1_curr, ker2_curr, ker1_next, ker2_next, length, tstep);

        velocity_verlet_scheme(phi1, phi2, phidot1, phidot2, ker1_curr, ker2_curr, ker1_next, ker2_next);

        float kappa = string_tension(t_evol);
        float R = scale_factor(t_evol);
        float time = physical_time(t_evol);

        if (should_count_strings(tstep, parameters.string_checks, final_step)) {

            // compute axion field: 
            for (int i = 0; i < length; i++) axion[i] = atan2(phi1[i], phi2[i]);

            if (parameters.NDIMS == 2) {
                int num_cores = Cores2D(axion, parameters.thr);
                float xi = num_cores * gsl_pow_2(time)/(gsl_pow_2(t_evol));
                printf("%f %f\n",time,xi);
            }
            if (parameters.NDIMS == 3) {
                int num_cores = Cores3D(axion, parameters.thr);
                float xi = num_cores * gsl_pow_2(time)/(gsl_pow_2(t_evol));
                printf("%f %f\n",time,xi);
            }
        }

        if (should_save_snapshot(tstep, parameters.n_snapshots, final_step)) {
            printf("Writing snapshot %d:\n", n_snapshots_written);

            // output time variables:
            printf("  string tension = %f\n", kappa);
            printf("  scale factor   = %f\n", R);
            printf("  time           = %f\n", time);

            // file names:
            char fname_phi1[50], fname_phi2[50];
            sprintf(fname_phi1, "phi1-snapshot%d", n_snapshots_written);
            sprintf(fname_phi2, "phi2-snapshot%d", n_snapshots_written);

            save_data(fname_phi1, phi1, length);
            save_data(fname_phi2, phi2, length);

            n_snapshots_written++;
        }
    }

    // Free memory:
    free(phi1);
    free(phi2);
    free(phidot1);
    free(phidot2);
    free(ker1_curr);
    free(ker2_curr);
    free(ker1_next);
    free(ker2_next);
}


void debug(dtype *phi1, dtype *phi2,
           dtype *phidot1, dtype *phidot2,
           dtype *ker1_curr, dtype *ker2_curr,
           dtype *ker1_next, dtype *ker2_next,
           int length, int tstep) {

    for (int i = 0; i < length; i++) {
        if (gsl_isnan(phi1[i])) {
            printf("NaN encountered in phi1\n");
            printf(" tstep = %d\n", tstep);
            assert(0);
        }
        if (gsl_isnan(phi2[i])) {
            printf("NaN encountered in phi2\n");
            printf(" tstep = %d\n", tstep);
            assert(0);
        }
        if (gsl_isnan(phidot1[i])) {
            printf("NaN encountered in phidot1\n");
            printf(" tstep = %d\n", tstep);
            assert(0);
        }
        if (gsl_isnan(phidot2[i])) {
            printf("NaN encountered in phidot2\n");
            printf(" tstep = %d\n", tstep);
            assert(0);
        }
        if (gsl_isnan(ker1_curr[i])) {
            printf("NaN encountered in ker1_curr\n");
            printf(" tstep = %d\n", tstep);
            assert(0);
        }
        if (gsl_isnan(ker2_curr[i])) {
            printf("NaN encountered in ker2_curr\n");
            printf(" tstep = %d\n", tstep);
            assert(0);
        }
        if (gsl_isnan(ker1_next[i])) {
            printf("NaN encountered in ker1_next\n");
            printf(" tstep = %d\n", tstep);
            assert(0);
        }
        if (gsl_isnan(ker2_next[i])) {
            printf("NaN encountered in ker2_next\n");
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