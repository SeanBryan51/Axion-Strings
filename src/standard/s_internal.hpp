#pragma once

#include "common/common.hpp"

// Mega struct containing all the pointers to large arrays/solution vectors:
typedef struct _all_data {
    data_t *phi1; // phi_1 field values
    data_t *phi2; // phi_2 field values
    data_t *phidot1; // phi_1 time derivative
    data_t *phidot2; // phi_2 time derivative
    data_t *ker1_curr; // current kernel for phi_1 equation of motion
    data_t *ker2_curr; // current kernel for phi_2 equation of motion
    data_t *ker1_next; // next kernel for phi_1 equation of motion
    data_t *ker2_next; // next kernel for phi_2 equation of motion

    data_t *axion; // axion field values
    data_t *saxion; // saxion field values

    sparse_matrix_t coefficient_matrix;

} all_data;

/**
 * Stencil coefficients: https://en.wikipedia.org/wiki/Finite_difference_coefficient
 */
inline data_t laplacian2D(data_t *phi, int i, int j, float dx, int N) {
    return (phi[offset2(i+1,j,N,0)] + phi[offset2(i,j+1,N,0)] - 4.0f*phi[offset2(i,j,N,0)] + phi[offset2(i-1,j,N,0)] + phi[offset2(i,j-1,N,0)]) / (pow_2(dx));
}

/**
 * Stencil coefficients: https://en.wikipedia.org/wiki/Finite_difference_coefficient
 */
inline data_t laplacian3D(data_t *phi, int i, int j, int k, float dx, int N) {
    return (phi[offset3(i+1,j,k,N)] + phi[offset3(i,j+1,k,N)] + phi[offset3(i,j,k+1,N)] - 6.0f*phi[offset3(i,j,k,N)] + phi[offset3(i-1,j,k,N)] + phi[offset3(i,j-1,k,N)] + phi[offset3(i,j,k-1,N)]) / pow_2(dx);
}

// s_integrate.cpp
void  build_coefficient_matrix(sparse_matrix_t *handle, int NDIMS, int N);
data_t laplacian2D(data_t *phi, int i, int j, float dx, int N);
data_t laplacian3D(data_t *phi, int i, int j, int k, float dx, int N);
void  vvsl_field_rescaled(all_data data);
void  vvsl_hamiltonian_form(all_data data);
