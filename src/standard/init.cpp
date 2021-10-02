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
    float L = parameters.space_step * N;

    // Set up rng stream:
    VSLStreamStatePtr stream;
    vslNewStream(&stream, VSL_BRNG_SFMT19937, parameters.seed);

    // Use input solution vectors to store random numbers for now:
    dtype *rng_array_1 = phi1;
    dtype *rng_array_2 = phi2;
    dtype *rng_array_3 = phidot1;
    dtype *rng_array_4 = phidot2;

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
            kx[i] = ky[i] = (- N / 2.0f + i) * 2.0f * M_PI / L;
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
            dtype k = sqrt(kx[i]*kx[i] + ky[j]*ky[j]);
            dtype amplitude, amplitude_dot;
            if (k != 0.0f) {
                dtype omegak = sqrt(pow_2(k) + m_eff_squared);
                dtype bose = 1.0f / (exp(omegak / T_initial) - 1.0f);
                amplitude = sqrt(bose / omegak); // Power spectrum for phi
                amplitude_dot = sqrt(bose * omegak); // Power spectrum for phidot
            } else {
                amplitude = 0.0f; // Choose average value of the fields to be zero.
                amplitude_dot = 0.0f;
            }

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
            // Remember to include normalisation of the dft.
            phi1[i] = phi[i][0] / pow_2(N); // Re(phi)
            phi2[i] = phi[i][1] / pow_2(N); // Im(phi)
            phidot1[i] = phidot[i][0] / pow_2(N); // Re(phidot)
            phidot2[i] = phidot[i][1] / pow_2(N); // Im(phidot)
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
            kx[i] = ky[i] = kz[i] = (- N / 2.0f + i) * 2.0f * M_PI / L;
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
            dtype k = sqrt(kx[i]*kx[i] + ky[j]*ky[j] + kz[l]*kz[l]);
            dtype amplitude, amplitude_dot;
            if (k != 0.0f) {
                dtype omegak = sqrt(pow_2(k) + m_eff_squared);
                dtype bose = 1.0f / (exp(omegak / T_initial) - 1.0f);
                amplitude = sqrt(bose / omegak); // Power spectrum for phi
                amplitude_dot = sqrt(bose * omegak); // Power spectrum for phidot
            } else {
                amplitude = 0.0f; // Choose average value of the fields to be zero.
                amplitude = 0.0f;
            }

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
            // Remember to include normalisation of the dft.
            phi1[i] = phi[i][0] / pow_3(N); // Re(phi)
            phi2[i] = phi[i][1] / pow_3(N); // Im(phi)
            phidot1[i] = phidot[i][0] / pow_3(N); // Re(phidot)
            phidot2[i] = phidot[i][1] / pow_3(N); // Im(phidot)
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

    fftw_cleanup_threads();
}