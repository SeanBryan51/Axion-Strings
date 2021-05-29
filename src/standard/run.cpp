#include <stdio.h>
#include <assert.h>
#include <gsl/gsl_math.h>

#include "../parameters.h"
#include "../utils/utils.h"

#include "common.h"
#include "interface.h"

int should_save_snapshot(int tstep, int n_snapshots, int final_tstep);
void debug(dtype *phi1, dtype *phi2, dtype *phidot1, dtype *phidot2, dtype *ker1_curr, dtype *ker2_curr, dtype *ker1_next, dtype *ker2_next, int length, int tstep);

void run_standard() {

    // Allocate fields on the heap:

    int length;
    assert(globals.NDIMS == 2 || globals.NDIMS == 3);
    if (globals.NDIMS == 3) {
        length = globals.N * globals.N * globals.N;
    } else {
        length = globals.N * globals.N;
    }

    dtype *phi1 = (dtype *) calloc(length, sizeof(dtype));
    dtype *phi2 = (dtype *) calloc(length, sizeof(dtype));
    dtype *phidot1 = (dtype *) calloc(length, sizeof(dtype));
    dtype *phidot2 = (dtype *) calloc(length, sizeof(dtype));

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
    int final_step = globals.light_time - round(1 / globals.DeltaRatio) + 1;
    for (int tstep = 0; tstep < final_step; tstep++) {

        debug(phi1, phi2, phidot1, phidot2, ker1_curr, ker2_curr, ker1_next, ker2_next, length, tstep);

        if (should_save_snapshot(tstep, globals.n_snapshots, final_step)) {
            // file names:
            char fname_phi1[50], fname_phi2[50];
            sprintf(fname_phi1, "phi1-snapshot%d", n_snapshots_written);
            sprintf(fname_phi2, "phi2-snapshot%d", n_snapshots_written);

            save_data(fname_phi1, phi1, length);
            save_data(fname_phi2, phi2, length);

            n_snapshots_written++;
        }

        // TIME EVOLUTON ALGORITHM (velocity-Verlet drift-kick algorithm, see Eq (125) TAOSTEU)

        for (int i = 0; i < length; i++) {
            phi1[i] += globals.dtau * (phidot1[i] + 0.5f * ker1_curr[i] * globals.dtau);
            phi2[i] += globals.dtau * (phidot2[i] + 0.5f * ker2_curr[i] * globals.dtau);
        }

        globals.t_evol = globals.t_evol + globals.dtau;

        kernels(ker1_next, ker2_next, phi1, phi2, phidot1, phidot2);

        for (int i = 0; i < length; i++) {
            phidot1[i] += 0.5f * (ker1_curr[i] + ker1_next[i]) * globals.dtau;
            phidot2[i] += 0.5f * (ker2_curr[i] + ker2_next[i]) * globals.dtau;
        }

        for (int i = 0; i < length; i++) {
            ker1_curr[i] = ker1_next[i];
            ker2_curr[i] = ker2_next[i];
        }


        // TODO: log this
        // TIME VARIABLES 

        // float kappa = log(globals.t_evol/globals.ms); // String tension (also time variable)
        // float R = globals.t_evol/(globals.ms*globals.L); // Scale factor in L units
        // float time = globals.t0 * powf(R/globals.R0*globals.ms, 2.0f); // Cosmic time in L units 

        // PHYSICAL FIELDS
        // TODO: complex numbers
        // PHI = phi1 + 1j * phi2
        // PHIDOT = phidot1 +1j * phidot2
        // axiondot = (PHIDOT/PHI).imag
        // adot_screen = saxion*axiondot
        // axion = arctan2(phi1,phi2) 
        // saxion = sqrt(phi1**2.0+phi2**2.0) 

        // STRINGS  
    
        // Nchecks = 100
        // to_analyse = arange(0, final_step,int(final_step/Nchecks))
        // thr = 1 # In % value
        // if tstep in to_analyse:
        //     num_cores = cores_pi(axion,N,thr)
        //     xi_2D = num_cores*time**2.0/(t_evol**2.0)
        //     mu = pi*kappa
        //     string_en = mu*xi_2D/time**2.0
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


void debug(dtype *phi1, dtype *phi2, dtype *phidot1, dtype *phidot2, dtype *ker1_curr, dtype *ker2_curr, dtype *ker1_next, dtype *ker2_next, int length, int tstep) {
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
    // TODO: temporary solution
    return tstep == 0 || tstep % (final_tstep / (n_snapshots - 1)) == 0;
}