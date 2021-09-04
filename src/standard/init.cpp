/*
 * Description: Module that sets the initial conditions for the string simulation
 */

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

void gaussian_thermal(dtype *phi1, dtype *phi2, dtype *phidot1, dtype *phidot2) {

    int length = get_length(), N = parameters.N;

    // Set up rng stream:
    VSLStreamStatePtr stream;
    vslNewStream(&stream, VSL_BRNG_SFMT19937, parameters.seed);

    dtype *rng_array_1 = (dtype *) calloc(length, sizeof(dtype));
    dtype *rng_array_2 = (dtype *) calloc(length, sizeof(dtype));
    dtype *rng_array_3 = (dtype *) calloc(length, sizeof(dtype));
    dtype *rng_array_4 = (dtype *) calloc(length, sizeof(dtype));

    assert(rng_array_1 != NULL && rng_array_2 != NULL && rng_array_3 != NULL && rng_array_4 != NULL);

    // Fill arrays with random numbers:
    mkl_v_rng_gaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, length, rng_array_1, 0.0f, 1.0f);
    mkl_v_rng_gaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, length, rng_array_2, 0.0f, 1.0f);
    mkl_v_rng_gaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, length, rng_array_3, 0.0f, 1.0f);
    mkl_v_rng_gaussian(VSL_RNG_METHOD_GAUSSIAN_ICDF, stream, length, rng_array_4, 0.0f, 1.0f);

    int status = fftw_init_threads();
    assert(status != 0);

    fftw_plan_with_nthreads(omp_get_max_threads());

    if (parameters.NDIMS == 2) {

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

        #pragma omp parallel for schedule(static)
        for (int m = 0; m < length; m++) {
            int i, j;
            coordinate2(&i, &j, m, N);

            dtype k = sqrt(kx[i]*kx[i] + ky[j]*ky[j] + 1e-10);
            dtype omegak = sqrt(pow_2(k * M_PI / N) + m_eff_squared);
            dtype bose = 1.0f / (exp(omegak / T_initial) - 1.0f);
            // TODO: okay to remove L normalisation inside sqrt?
            dtype amplitude = sqrt(bose / omegak); // Power spectrum for phi
            dtype amplitude_dot = sqrt(bose * omegak); // Power spectrum for phidot

            phi_k[m][0] = amplitude * rng_array_1[m];
            phi_k[m][1] = amplitude * rng_array_2[m];
            phidot_k[m][0] = amplitude_dot * rng_array_3[m];
            phidot_k[m][1] = amplitude_dot * rng_array_4[m];
        }

        fftw_execute(p1);
        fftw_execute(p2);

        shift2D(phi, N);
        shift2D(phidot, N);

        #pragma omp parallel for schedule(static)
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

        #pragma omp parallel for schedule(static)
        for (int m = 0; m < length; m++) {
            int i, j, l;
            coordinate3(&i, &j, &l, m, N);
            dtype k = sqrt(kx[i]*kx[i] + ky[j]*ky[j] + kz[l]*kz[l] + 1e-10);
            dtype omegak = sqrt(pow_2(k * M_PI / N) + m_eff_squared);
            dtype bose = 1.0f / (exp(omegak / T_initial) - 1.0f);
            // TODO: okay to remove L normalisation inside sqrt?
            dtype amplitude = sqrt(bose / omegak); // Power spectrum for phi
            dtype amplitude_dot = sqrt(bose * omegak); // Power spectrum for phidot

            phi_k[m][0] = amplitude * rng_array_1[m];
            phi_k[m][1] = amplitude * rng_array_2[m];
            phidot_k[m][0] = amplitude_dot * rng_array_3[m];
            phidot_k[m][1] = amplitude_dot * rng_array_4[m];
        }

        fftw_execute(p1);
        fftw_execute(p2);

        shift3D(phi, N);
        shift3D(phidot, N);

        #pragma omp parallel for schedule(static)
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
    #pragma omp parallel for schedule(static) reduction(+:phi1_mean,phi2_mean,phidot1_mean,phidot2_mean)
    for (int i = 0; i < length; i++) {
        phi1_mean += phi1[i] / length;
        phi2_mean += phi2[i] / length;
        phidot1_mean += phidot1[i] / length;
        phidot2_mean += phidot2[i] / length;
    }

    // 2. Calculate standard deviation.
    dtype phi1_sd, phi2_sd, phidot1_sd, phidot2_sd;
    phi1_sd = phi2_sd = phidot1_sd = phidot2_sd = 0.0f;
    #pragma omp parallel for schedule(static) reduction(+:phi1_sd,phi2_sd,phidot1_sd,phidot2_sd)
    for (int i = 0; i < length; i++) {
        phi1_sd += pow_2(phi1[i] - phi1_mean) / (length - 1.0f);
        phi2_sd += pow_2(phi2[i] - phi2_mean) / (length - 1.0f);
        phidot1_sd += pow_2(phidot1[i] - phidot1_mean) / (length - 1.0f);
        phidot2_sd += pow_2(phidot2[i] - phidot2_mean) / (length - 1.0f);
    }

    // TODO: uncommenting the following causes the simulation to diverge! WHY
    // phi1_sd = pow(phi1_sd, 0.5);
    // phi2_sd = pow(phi2_sd, 0.5);
    // phidot1_sd = pow(phidot1_sd, 0.5);
    // phidot2_sd = pow(phidot2_sd, 0.5);

    // 3. Normalise.
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < length; i++) {
        phi1[i] = (phi1[i] - phi1_mean) / phi1_sd;
        phi2[i] = (phi2[i] - phi2_mean) / phi2_sd;
        phidot1[i] = (phidot1[i] - phidot1_mean) / phidot1_sd;
        phidot2[i] = (phidot2[i] - phidot2_mean) / phidot2_sd;
    }

    fftw_cleanup_threads();
}