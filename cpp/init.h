#pragma once

// function declarations:
void init_noise(float * phi1, float * phi2, float * phi1dot, float *phi2dot);
void gaussian_thermal(float * phi1, float * phi2, float * phi1dot, float *phi2dot, bool flag_normalise);