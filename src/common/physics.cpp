#include "common.hpp"

data_t tau;
float T_initial;
float reduced_planck_mass;
float m_saxion_initial;
float g_star;
float m_eff_squared;
float light_crossing_time;

/*
 * Initialise all physical parameters / constants.
 */
void set_physics_variables() {

    // Saxion mass in units of f_a: m_saxion = sqrt(lambda) * f_a / f_a
    m_saxion_initial = sqrtf(parameters.lambdaPRS);

    // Initial temperature in units of f_a.
    // Note: only need to set T to some value greater than sqrt(3)*(fa)
    // to simulate the PQ phase transition.
    T_initial = 4.0f;

    // Effective mass of the PQ potential: m_eff^2 = lambda ( T^2/3 - fa^2 )
    m_eff_squared = parameters.lambdaPRS * (pow_2(T_initial) / 3.0f - 1.0f);

    // Light crossing time: approximate time for light to travel one Hubble volume,
    // i.e. when H^{-1} ~ L.
    light_crossing_time = 0.5f * parameters.N * parameters.space_step;

    // Dimensionless program time variable (in conformal time).
    tau = 1.0f;
}

/*
 * Dimnensionless physical time (in program units).
 */
float physical_time() {
    return 0.5f * pow_2(tau); // Found from integratating definition of dimensionless tau and imposing that tau_0 = t_0 = 0.
}

float scale_factor() {
    // Radiation dominated background: scale factor increases linearly with conformal time.
    return tau;
}

/*
 * Dimensionless Hubble parameter: \tilde{H} \equiv H / f_a
 * Note: H is normalised with respect to its initial value H / H_0.
 */
float hubble_parameter() {
    return 1.0f / pow_2(tau);
}

/*
 * Temperature in units of f_a.
 */
float temperature() {
    return T_initial / (tau);
}

/*
 * String tension mu / mu_0, where mu_0 = \pi f_a^2
 */
float string_tension() {
    return log(m_saxion_initial / hubble_parameter());
}

/*
 * Effective mass of PQ potential.
 */
float meff_squared() {
    return parameters.lambdaPRS * (pow_2(temperature()) / 3.0f - 1.0f);
}
