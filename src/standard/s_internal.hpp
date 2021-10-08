#pragma once

#include "common/common.hpp"

// Mega struct containing all the pointers to large arrays/solution vectors:
struct _all_data {
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

};

// s_integrate.cpp
void  build_coefficient_matrix(sparse_matrix_t *handle, int NDIMS, int N);
data_t laplacian2D(data_t *phi, int i, int j, float dx, int N);
data_t laplacian3D(data_t *phi, int i, int j, int k, float dx, int N);
void  vvsl_field_rescaled(all_data data);
void  vvsl_hamiltonian_form(all_data data);
