#include <stdio.h>
#include <stdlib.h>
#include <gsl/gsl_math.h>
#include <assert.h>

#include "standard/spatial.h"
#include "utils/utils.h"
#include "parameters.h"

#define MAX_STENCIL 2

int is_equal(float x, float y, float tolerance) {
    assert(tolerance > 0.0f);
    return x - y <= tolerance && x - y >= -tolerance;
}

void test_laplacian2D() {

    // success case: simple array of zeros
    int N = 128;
    float *phi = (float *) calloc(N * N, sizeof(float));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            assert(is_equal(laplacian2D(phi, i, j, 1.0f, N), 0.0f, 1e-30f));
        }
    }
    free(phi);

    // success case: wave with periodic boundary conditions
    N = 256;
    phi = (float *) calloc(N * N, sizeof(float));
    float L = N / 2.0f;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            // define field values
            phi[offset2(i,j,N)] = sinf(M_PI / L * i) + cosf(M_PI / L * j);
        }
    }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            // compare discretised laplacian and actual laplacian
            float discretised_laplacian = laplacian2D(phi, i, j, 1.0f, N);
            float actual_laplacian = - gsl_pow_2(M_PI / L) * (sinf(M_PI / L * i) + cosf(M_PI / L * j));
            assert(is_equal(discretised_laplacian, actual_laplacian, 1e-5f));
        }
    }
    free(phi);
}

void test_laplacian3D() {

    // success case: simple array of zeros
    int N = 128;
    float *phi = (float *) calloc(N * N * N, sizeof(float));
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                assert(is_equal(laplacian3D(phi, i, j, k, 1.0f, N), 0.0f, 1e-5f));
            }
        }
    }
    free(phi);

    // success case: wave with periodic boundary conditions
    N = 256;
    phi = (float *) calloc(N * N * N, sizeof(float));
    float L = N / 2.0f;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                // define field values
                phi[offset3(i,j,k,N)] = sinf(M_PI / L * i) + cosf(M_PI / L * j) + sinf(M_PI / L * k);
            }
        }
    }
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < N; k++) {
                // compare discretised laplacian and actual laplacian
                float discretised_laplacian = laplacian3D(phi, i, j, k, 1.0f, N);
                float actual_laplacian = - gsl_pow_2(M_PI / L) * (sinf(M_PI / L * i) + cosf(M_PI / L * j) + sinf(M_PI / L * k));
                assert(is_equal(discretised_laplacian, actual_laplacian, 1e-5f));
            }
        }
    }
    free(phi);
}

int main(void) {
    initialise_globals((char *) "/Users/seanbryan/Documents/UNI/2021T1/Project/Axion-Strings/src/tests/test_parameters.param");
    // Test all stencil settings:
    for (globals.StencilOrder = 2; globals.StencilOrder <= MAX_STENCIL; globals.StencilOrder += 2) {
        test_laplacian2D();
        test_laplacian3D();
    }
    return EXIT_SUCCESS;
}