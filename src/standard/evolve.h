#pragma once

void kernels(float *K1, float *K2, float *phi1, float *phi2, float *phidot1, float *phidot2);
void apply_drift(float *phi1, float *phi2, float *phi1dot, float *phi2dot, float *K1, float *K2);
void apply_kick(float *phi1dot, float *phi2dot, float *K1, float*K2, float *K1_next, float *K2_next);