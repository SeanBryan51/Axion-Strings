#pragma once

// function declarations:

void init_noise(float * phi1, float * phi2, float * phidot1, float *phidot2);
void gaussian_thermal(float * phi1, float * phi2, float * phidot1, float *phidot2, int flag_normalise);