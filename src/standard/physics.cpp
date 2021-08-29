#include "common.h"

float t_evol;
float t_initial;
float T_initial;
float t_phys_initial;
float R_initial;
float reduced_planck_mass;
float m_saxion;
float g_star;
float m_eff_squared;
float light_crossing_time;

/*
 * Initialise all physical parameters / constants.
 */
void set_internal_variables() {

    // Reduced Planck mass in GeV normalised by the axion decay constant f_a: M_planck = 1 / sqrt(8*pi*G) / f_a
    reduced_planck_mass = 2.4f * 1e18 / parameters.fa_phys;

    // Saxion mass in units of f_a: m_saxion = sqrt(lambda) * f_a / f_a
    m_saxion = sqrtf(parameters.lambdaPRS);

    // Relativistic degrees of freedom:
    g_star = 106.75f;

    // Initial conformal time.
    t_initial = parameters.space_step - parameters.time_step;

    // Initial temperature in units of f_a. Defined when H ~ f_a
    // T_initial = powf(90.0f * gsl_pow_2(reduced_planck_mass) / (g_star * M_PI * M_PI), 0.25f);
    // TODO: the simulation breaks when T >> 1, we only need to set T to some
    // value greater than sqrt(3)*(fa) to simulate the PQ phase transition.
    T_initial = 4.0f;

    // Initial physical time.
    // TODO: why is t0 = t_phys_initial defined as Delta / (2.0f * L * ms * ms) ?
    t_phys_initial = 1.0f / (2.0f * parameters.N * m_saxion * m_saxion);

    // Initial scale factor.
    // TODO: should R0 = R_initial be (Delta * ms / (ms * L)) = 1 / N 
    // or (1.0 / sqrtf(parameters.fa_phys / m_saxion)) ?
    R_initial = 1.0f / parameters.N;

    // Effective mass of the PQ potential: m_eff^2 = lambda ( T^2/3 - fa^2 )
    m_eff_squared = parameters.lambdaPRS * (gsl_pow_2(T_initial) / 3.0f - 1.0f);

    // Light crossing time: approximate time for light to travel one Hubble volume.
    light_crossing_time = 0.5f * parameters.N * parameters.space_step;

    // Dimensionless program time variable (in conformal time).
    t_evol = t_initial;
}

float physical_time(float t_conformal) {
    // float time = parameters.t0 * powf(R/parameters.R0*parameters.ms, 2.0f); // Cosmic time in L units 
    float R = scale_factor(t_conformal);
    return t_phys_initial * powf(R/R_initial*m_saxion, 2.0f); // Cosmic time in L units
    // return t_phys_initial * powf(t_conformal/t_initial, 2.0f);
}

float scale_factor(float t_conformal) {
    // In a radiation dominated background the scale factor increases
    // linearly with conformal time.
    // TODO: ask Giovanni about why scale factor is defined as
    // return t_conformal/(parameters.ms*parameters.L); // Scale factor in L units
    return t_conformal/(m_saxion*parameters.N*parameters.space_step); // Scale factor in L units
    // return R_initial * (t_conformal / t_initial);
}

float hubble_parameter(float t_conformal) {
    // TODO: need to choose a initial reference value for the Hubble parameter at t_evol = t_initial
    return 0.0f;
}

float temperature(float t_conformal) {
    return T_initial / (t_conformal / t_initial);
}

float string_tension(float t_conformal) {
    return log(t_conformal/m_saxion); // String tension (also time variable)
}

float meff_squared(float t_conformal) {
    return parameters.lambdaPRS * (gsl_pow_2(temperature(t_conformal)) / 3.0f - 1.0f);
}
