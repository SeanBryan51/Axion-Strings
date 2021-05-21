#pragma once

#include <stdlib.h>

#define MAX_LEN 100

extern struct _globals {

    // User defined parameters to be read from 
    // parameter file:

    int   NDIMS;
    int   N;
    float Delta;
    float DeltaRatio;
    int   StencilOrder;
    float fa_phys;
    float lambdaPRS;
    char  Potential[MAX_LEN];
    float Era;

    int   save_snapshots;
    int   n_snapshots;
    char  output_directory[MAX_LEN];

    unsigned int seed;

    // Independent parameters:
    // TODO: remove independent (derived) parameters and declare when we need them
    // for clarity?

    float Mpl;
    float fa;
    float ms;
    float L;
    float Delta_tau;
    float H1;

    // Code spacings:
    float dx;
    float dtau;
    float t_evol;
    int   light_time;
    float gstar;
    float T0;
    float R0;
    float t0;
    float meffsquared;

} globals;
