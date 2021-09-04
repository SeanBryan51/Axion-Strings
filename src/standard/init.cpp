/*
 * Description: Module that sets the initial conditions for the string simulation
 */

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_math.h>
#include <fftw3.h>

#include "common.h"

/*
 * See question 3.11. from http://www.fftw.org/faq/
 */
static void shift2D(fftw_complex *arr, int N) {
    int length = get_length();
    #pragma omp parallel for schedule(static)
    for (int m = 0; m < length; m++) {
        int i, j;
        coordinate2(&i, &j, m, N);
        arr[m][0] *= pow(-1.0f, i + j);
        arr[m][1] *= pow(-1.0f, i + j);
    }
}

/*
 * See question 3.11. from http://www.fftw.org/faq/
 */
static void shift3D(fftw_complex *arr, int N) {
    int length = get_length();
    #pragma omp parallel for schedule(static)
    for (int m = 0; m < length; m++) {
        int i, j, k;
        coordinate3(&i, &j, &k, m, N);
        arr[m][0] *= pow(-1.0f, i + j + k);
        arr[m][1] *= pow(-1.0f, i + j + k);
    }
}

/* 
 * Does the following:
 * 1. Allocates memory for solution vectors in all_data struct.
 * 2. Initialises physical parameters defined in physics.cpp.
 */
void initialise_data(all_data *data) {

    int length = get_length();

    // Allocate fields on the heap:

    dtype *phi1 = (dtype *) calloc(length, sizeof(dtype));
    dtype *phi2 = (dtype *) calloc(length, sizeof(dtype));
    dtype *phidot1 = (dtype *) calloc(length, sizeof(dtype));
    dtype *phidot2 = (dtype *) calloc(length, sizeof(dtype));

    // Assert allocation was successful:
    assert(phi1 != NULL && phi2 != NULL && phidot1 != NULL && phidot2 != NULL);

    // Initialise fields:
    // TODO: remove init_noise()
    // init_noise(phi1, phi2, phidot1, phidot2);
    gaussian_thermal(phi1, phi2, phidot1, phidot2);

    // Allocate field kernels on the heap:
    dtype *ker1_curr = (dtype *) calloc(length, sizeof(dtype));
    dtype *ker2_curr = (dtype *) calloc(length, sizeof(dtype));
    dtype *ker1_next = (dtype *) calloc(length, sizeof(dtype));
    dtype *ker2_next = (dtype *) calloc(length, sizeof(dtype));

    // Assert allocation was successful:
    assert(ker1_curr != NULL && ker2_curr != NULL && ker1_next != NULL && ker2_next != NULL);

    dtype *axion = NULL;
    if (parameters.run_string_finding || parameters.save_snapshots) axion = (dtype *) calloc(length, sizeof(dtype));

    dtype *saxion = NULL;

    data->phi1 = phi1;
    data->phi2 = phi2;
    data->phidot1 = phidot1;
    data->phidot2 = phidot2;
    data->ker1_curr = ker1_curr;
    data->ker2_curr = ker2_curr;
    data->ker1_next = ker1_next;
    data->ker2_next = ker2_next;
    data->axion = axion;
    data->saxion = saxion;

    // Initialise kernels for the next time step:
    kernels(data->ker1_next, data->ker2_next, *data);
}

/*
 * Free memory allocated in all_data struct.
 */
void free_all_data(all_data data) {
    free(data.phi1);
    free(data.phi2);
    free(data.phidot1);
    free(data.phidot2);
    free(data.ker1_curr);
    free(data.ker2_curr);
    free(data.ker1_next);
    free(data.ker2_next);
}

/*
 * Random white noise in position space, independent of the shape
 * of the potential.
 */
void init_noise(dtype *phi1, dtype *phi2, dtype *phidot1, dtype *phidot2) {

    int N = parameters.N;
    gsl_rng *rng = gsl_rng_alloc(gsl_rng_ranlxs0);
    gsl_rng_set(rng, parameters.seed);

    dtype th, r;
    if (parameters.NDIMS == 2) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                th = 2 * M_PI * gsl_rng_uniform(rng);
                r = gsl_ran_gaussian(rng, 0.1f) + 1.0f;
                // Note: offset(x,y) = (x + ny * y)
                phi1[offset2(i,j,N)] = r * cosf(th);
                phi2[offset2(i,j,N)] = r * sinf(th);
                phidot1[offset2(i,j,N)] = phidot2[offset2(i,j,N)] = 0;
            }
        }
    } else if (parameters.NDIMS == 3) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                for (int k = 0; k < N; k++) {
                    th = 2 * M_PI * gsl_rng_uniform(rng);
                    r = gsl_ran_gaussian(rng, 0.1f) + 1.0f;
                    // Note: offset(x,y,z) = (x * ny + y) * nz + z
                    phi1[offset3(i,j,k,N)] = r * cosf(th);
                    phi2[offset3(i,j,k,N)] = r * sinf(th);
                    phidot1[offset3(i,j,k,N)] = phidot2[offset3(i,j,k,N)] = 0;
                }
            }
        }
    }
}

void gaussian_thermal(dtype * phi1, dtype * phi2, dtype * phidot1, dtype *phidot2) {

    int length, N = parameters.N;
    gsl_rng *rng = gsl_rng_alloc(gsl_rng_ranlxs0);
    gsl_rng_set(rng, parameters.seed);

    if (parameters.NDIMS == 2) {
        length = N * N;

        dtype *kx = (dtype *) calloc(N, sizeof(dtype));
        dtype *ky = (dtype *) calloc(N, sizeof(dtype));

        assert(kx != NULL && ky != NULL);

        for(int i = 0; i < N; i++) {
            kx[i] = ky[i] = - N / 2.0f + i;
        }

        fftw_complex *phi_k, *phi, *phidot_k, *phidot;
        fftw_plan p1, p2;

        phi_k = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * length);
        phi = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * length);
        p1 = fftw_plan_dft_2d(N, N, phi_k, phi, FFTW_BACKWARD, FFTW_ESTIMATE);

        phidot_k = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * length);
        phidot = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * length);
        p2 = fftw_plan_dft_2d(N, N, phidot_k, phidot, FFTW_BACKWARD, FFTW_ESTIMATE);

#if 0
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {

                dtype k = sqrt(kx[i]*kx[i] + ky[j]*ky[j] + 1e-10);
                dtype omegak = sqrt(gsl_pow_2(k * M_PI / N) + m_eff_squared);
                dtype bose = 1.0f / (exp(omegak / T_initial) - 1.0f);
                // TODO: okay to remove L normalisation inside sqrt?
                dtype amplitude = sqrt(bose / omegak); // Power spectrum for phi
                dtype amplitude_dot = sqrt(bose * omegak); // Power spectrum for phidot

                phi_k[offset2(i,j,N)][0] = amplitude * gsl_ran_gaussian(rng, 1.0f);
                phi_k[offset2(i,j,N)][1] = amplitude * gsl_ran_gaussian(rng, 1.0f);
                phidot_k[offset2(i,j,N)][0] = amplitude_dot * gsl_ran_gaussian(rng, 1.0f);
                phidot_k[offset2(i,j,N)][1] = amplitude_dot * gsl_ran_gaussian(rng, 1.0f);

            }
        }
#endif
        // Can't parallelise with gsl
        // #pragma omp parallel for schedule(static)
        for (int m = 0; m < length; m++) {
            int i, j;
            coordinate2(&i, &j, m, N);

            dtype k = sqrt(kx[i]*kx[i] + ky[j]*ky[j] + 1e-10);
            dtype omegak = sqrt(pow_2(k * M_PI / N) + m_eff_squared);
            dtype bose = 1.0f / (exp(omegak / T_initial) - 1.0f);
            // TODO: okay to remove L normalisation inside sqrt?
            dtype amplitude = sqrt(bose / omegak); // Power spectrum for phi
            dtype amplitude_dot = sqrt(bose * omegak); // Power spectrum for phidot

            phi_k[m][0] = amplitude * gsl_ran_gaussian(rng, 1.0f);
            phi_k[m][1] = amplitude * gsl_ran_gaussian(rng, 1.0f);
            phidot_k[m][0] = amplitude_dot * gsl_ran_gaussian(rng, 1.0f);
            phidot_k[m][1] = amplitude_dot * gsl_ran_gaussian(rng, 1.0f);
        }

        fftw_execute(p1);
        fftw_execute(p2);

        shift2D(phi, N);
        shift2D(phidot, N);

        for (int i = 0; i < length; i++) {
            phi1[i] = phi[i][0]; // Re(phi)
            phi2[i] = phi[i][1]; // Im(phi)
            phidot1[i] = phidot[i][0]; // Re(phidot)
            phidot2[i] = phidot[i][1]; // Im(phidot)
        }

        fftw_destroy_plan(p1);
        fftw_free(phi_k);
        fftw_free(phi);
        fftw_destroy_plan(p2);
        fftw_free(phidot_k);
        fftw_free(phidot);
        free(kx);
        free(ky);
    }

    if (parameters.NDIMS == 3) {
        length = N * N * N;

        dtype *kx = (dtype *) calloc(N, sizeof(dtype));
        dtype *ky = (dtype *) calloc(N, sizeof(dtype));
        dtype *kz = (dtype *) calloc(N, sizeof(dtype));

        assert(kx != NULL && ky != NULL && kz != NULL);

        for(int i = 0; i < N; i++) {
            kx[i] = ky[i] = kz[i] = - N / 2.0f + i;
        }

        fftw_complex *phi_k, *phi, *phidot_k, *phidot;
        fftw_plan p1, p2;

        phi_k = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * length);
        phi = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * length);
        p1 = fftw_plan_dft_3d(N, N, N, phi_k, phi, FFTW_BACKWARD, FFTW_ESTIMATE);

        phidot_k = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * length);
        phidot = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * length);
        p2 = fftw_plan_dft_3d(N, N, N, phidot_k, phidot, FFTW_BACKWARD, FFTW_ESTIMATE);

#if 0
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                for (int l = 0; l < N; l++) {

                    dtype k = sqrt(kx[i]*kx[i] + ky[j]*ky[j] + kz[l]*kz[l] + 1e-10);
                    dtype omegak = sqrt(gsl_pow_2(k * M_PI / N) + m_eff_squared);
                    dtype bose = 1.0f / (exp(omegak / T_initial) - 1.0f);
                    // TODO: okay to remove L normalisation inside sqrt?
                    dtype amplitude = sqrt(bose / omegak); // Power spectrum for phi
                    dtype amplitude_dot = sqrt(bose * omegak); // Power spectrum for phidot

                    phi_k[offset3(i,j,l,N)][0] = amplitude * gsl_ran_gaussian(rng, 1.0f);
                    phi_k[offset3(i,j,l,N)][1] = amplitude * gsl_ran_gaussian(rng, 1.0f);
                    phidot_k[offset3(i,j,l,N)][0] = amplitude_dot * gsl_ran_gaussian(rng, 1.0f);
                    phidot_k[offset3(i,j,l,N)][1] = amplitude_dot * gsl_ran_gaussian(rng, 1.0f);

                }
            }
        }
#endif
        // #pragma omp parallel for schedule(static)
        for (int m = 0; m < length; m++) {
            int i, j, l;
            coordinate3(&i, &j, &l, m, N);
            dtype k = sqrt(kx[i]*kx[i] + ky[j]*ky[j] + kz[l]*kz[l] + 1e-10);
            dtype omegak = sqrt(pow_2(k * M_PI / N) + m_eff_squared);
            dtype bose = 1.0f / (exp(omegak / T_initial) - 1.0f);
            // TODO: okay to remove L normalisation inside sqrt?
            dtype amplitude = sqrt(bose / omegak); // Power spectrum for phi
            dtype amplitude_dot = sqrt(bose * omegak); // Power spectrum for phidot

            phi_k[m][0] = amplitude * gsl_ran_gaussian(rng, 1.0f);
            phi_k[m][1] = amplitude * gsl_ran_gaussian(rng, 1.0f);
            phidot_k[m][0] = amplitude_dot * gsl_ran_gaussian(rng, 1.0f);
            phidot_k[m][1] = amplitude_dot * gsl_ran_gaussian(rng, 1.0f);
        }

        fftw_execute(p1);
        fftw_execute(p2);

        shift3D(phi, N);
        shift3D(phidot, N);

        for (int i = 0; i < length; i++) {
            phi1[i] = phi[i][0]; // Re(phi)
            phi2[i] = phi[i][1]; // Im(phi)
            phidot1[i] = phidot[i][0]; // Re(phidot)
            phidot2[i] = phidot[i][1]; // Im(phidot)
        }

        fftw_destroy_plan(p1);
        fftw_free(phi_k);
        fftw_free(phi);
        fftw_destroy_plan(p2);
        fftw_free(phidot_k);
        fftw_free(phidot);
        free(kx);
        free(ky);
        free(kz);
    }

    // TODO: what order of magnitude should the fluctuations in the field be?

    // 1. Calculate mean.
    dtype phi1_mean, phi2_mean, phidot1_mean, phidot2_mean;
    phi1_mean = phi2_mean = phidot1_mean = phidot2_mean = 0.0f;
    for (int i = 0; i < length; i++) {
        phi1_mean += phi1[i] / length;
        phi2_mean += phi2[i] / length;
        phidot1_mean += phidot1[i] / length;
        phidot2_mean += phidot2[i] / length;
    }

    // 2. Calculate standard deviation.
    dtype phi1_sd, phi2_sd, phidot1_sd, phidot2_sd;
    phi1_sd = phi2_sd = phidot1_sd = phidot2_sd = 0.0f;
    for (int i = 0; i < length; i++) {
        phi1_sd += gsl_pow_2(phi1[i] - phi1_mean) / (length - 1.0f);
        phi2_sd += gsl_pow_2(phi2[i] - phi2_mean) / (length - 1.0f);
        phidot1_sd += gsl_pow_2(phidot1[i] - phidot1_mean) / (length - 1.0f);
        phidot2_sd += gsl_pow_2(phidot2[i] - phidot2_mean) / (length - 1.0f);
    }

    // TODO: uncommenting the following causes the simulation to diverge! WHY
    // phi1_sd = pow(phi1_sd, 0.5);
    // phi2_sd = pow(phi2_sd, 0.5);
    // phidot1_sd = pow(phidot1_sd, 0.5);
    // phidot2_sd = pow(phidot2_sd, 0.5);

    // 3. Normalise.
    for (int i = 0; i < length; i++) {
        phi1[i] = (phi1[i] - phi1_mean) / phi1_sd;
        phi2[i] = (phi2[i] - phi2_mean) / phi2_sd;
        phidot1[i] = (phidot1[i] - phidot1_mean) / phidot1_sd;
        phidot2[i] = (phidot2[i] - phidot2_mean) / phidot2_sd;
    }
}