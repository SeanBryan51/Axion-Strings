#pragma once

#include <math.h>

//////////////////////////////////////////////////////////////////////////////
// SIMULATION PARAMETERS TO CHOOSE
//////////////////////////////////////////////////////////////////////////////

#define NDIMS 2 // Spatial dimensions of the simulation box. Set NDIMS = 2 or 3

const int   N = 1024; // Number of grid points
const float Delta = 1.0f; // Default Delta=1 and H=fa
const float DeltaRatio = 1.0f / 3.0f; // Time/Space step ratio
const int   StencilOrder = 2; // Choose order accuracy in dx for spatial derivative calculations. Options: StencilOrder = 2,4,6
const float fa_phys = 1e12; // Physical value of fa in GeV
const float lambdaPRS = 1.0f; // Quartic coupling 
const char* Potential = "Thermal"; // Choose potential. Options: 'MexicanHat' or 'Thermal'
const float Era = 1.0f; // era=1 for radiation domination, era=2 for early matter domination (in PRS trick)


// TO IMPLEMENT

// const float alpha = 1.0f // PRS fudge factor
// const int   nu = 1 // Choose simulation type. Options: nu = 0 (Physical), nu = 1 (PRS)



//////////////////////////////////////////////////////////////////////////////
// INDEPENDENT PARAMETERS 
//////////////////////////////////////////////////////////////////////////////

const float Mpl = 2.4f * 1e18 / fa_phys; // Normalized to fa_phys chosen 
const float fa = 1.0f; // In fa units
const float ms = sqrt(lambdaPRS) * fa; // Saxion mass for PRS in ADM units
const float L = Delta * N; // Comoving 
const float Delta_tau = DeltaRatio * Delta;
const float H1 = fa / Delta;


// Code spacings
const float dx = ms * Delta;
const float dtau = ms * Delta_tau;
float       t_evol = dtau / DeltaRatio - dtau; // Such that in the first loop iteration, t_evol = 1 
const int   light_time = round(0.5f * N / DeltaRatio);
const float gstar = 106.75f;
const float T0 = sqrt(Mpl * 90.0f / (gstar * M_PI * M_PI)); // In units of fa
const float R0 = Delta * ms / (ms * L);
const float t0 = Delta / (2.0f * L * ms * ms);
const float meffsquared = lambdaPRS * T0 * T0 / 3.0f;
