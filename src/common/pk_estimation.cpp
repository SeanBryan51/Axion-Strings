#include "common.hpp"

#include <fftw3.h>

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

/**
 * 
 * For a real field just set @param data_imag to NULL.
 */
void output_powerspec(char *file_name, data_t *data_real, data_t *data_imag) {

    int N = parameters.N;
    int NDIMS = parameters.NDIMS;
    float L = N * parameters.space_step;

    int length = get_length();

    int n_bins = N / 2 - 1; // N/2 - 1 because we ignore the k = 0 mode.
    data_t *pk = (data_t *) calloc(n_bins, sizeof(data_t));
    data_t *ks = (data_t *) calloc(n_bins, sizeof(data_t));
    int *count = (int *) calloc(n_bins, sizeof(int)); // count the number of modes in each bin.
    assert(pk != NULL && ks != NULL && count != NULL);

    fftw_plan_with_nthreads(omp_get_max_threads());

    if (NDIMS == 2) {

        data_t *kx = (data_t *) calloc(N, sizeof(data_t));
        data_t *ky = (data_t *) calloc(N, sizeof(data_t));
        assert(kx != NULL && ky != NULL);

        for (int i = 0; i < N; i++) kx[i] = ky[i] = (- N / 2.0f + i);

        fftw_complex *data, *data_k;
        fftw_plan plan;

        data   = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * length);
        data_k = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * length);
        plan = fftw_plan_dft_2d(N, N, data, data_k, FFTW_FORWARD, FFTW_ESTIMATE);

        #pragma omp parallel for schedule(static)
        for (int m = 0; m < length; m++) {
            data[m][0] = data_real[m];
            data[m][1] = (data_imag != NULL) ? data_imag[m] : 0.0f;
        }

        shift2D(data, N);

        fftw_execute(plan);

        #pragma omp parallel for schedule(static)
        for (int m = 0; m < length; m++) {
            int i, j;
            coordinate2(&i, &j, m, N);

            data_t kmag = sqrt(pow_2(kx[i]) + pow_2(ky[j]));
            if (kmag == 0.0f || kmag > N/2) continue; // ignore zero mode (field average) and all modes k > N/2

            int ps_index = floor(kmag) - 1;
            pk[ps_index] += pow_2(data_k[m][0]) + pow_2(data_k[m][1]);
            ks[ps_index] += kmag;
            count[ps_index]++;
        }

        for (int m = 0; m < n_bins; m++) {
            ks[m] /= count[m]; // compute average k value within bin.
            pk[m] /= 2.0f * M_PI * ks[m]; // angular average over '2-dimensional shell'
            ks[m] *= 2.0f * M_PI / L; // convert to dimensionful k.
        }

        fftw_destroy_plan(plan);
        fftw_free(data);
        fftw_free(data_k);
        free(kx);
        free(ky);
    }

    if (NDIMS == 3) {

        data_t *kx = (data_t *) calloc(N, sizeof(data_t));
        data_t *ky = (data_t *) calloc(N, sizeof(data_t));
        data_t *kz = (data_t *) calloc(N, sizeof(data_t));
        assert(kx != NULL && ky != NULL && kz != NULL);

        for (int i = 0; i < N; i++) kx[i] = ky[i] = kz[i] = (- N / 2.0f + i);

        fftw_complex *data, *data_k;
        fftw_plan plan;

        data   = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * length);
        data_k = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * length);
        plan = fftw_plan_dft_3d(N, N, N, data, data_k, FFTW_FORWARD, FFTW_ESTIMATE);

        shift3D(data, N);

        fftw_execute(plan);

        #pragma omp parallel for schedule(static)
        for (int m = 0; m < length; m++) {
            int i, j, k;
            coordinate3(&i, &j, &k, m, N);

            data_t kmag = sqrt(pow_2(kx[i]) + pow_2(ky[j]) + pow_2(kz[k]));
            if (kmag == 0.0f || kmag > N/2) continue; // ignore zero mode (field average) and all modes k > N/2

            int ps_index = floor(kmag) - 1;
            pk[ps_index] += pow_2(data_k[m][0]) + pow_2(data_k[m][1]);
            ks[ps_index] += kmag;
            count[ps_index]++;
        }

        for (int m = 0; m < n_bins; m++) {
            ks[m] /= count[m]; // compute average k value within bin.
            pk[m] /= 4.0f * M_PI * pow_2(ks[m]); // angular average over '3-dimensional shell'
            ks[m] *= 2.0f * M_PI / L; // convert to dimensionful k.
        }

        fftw_destroy_plan(plan);
        fftw_free(data);
        fftw_free(data_k);
        free(kx);
        free(ky);
        free(kz);
    }

    fio_save_pk(file_name, pk, ks, count, n_bins);

    free(pk);
    free(ks);
    free(count);
}
