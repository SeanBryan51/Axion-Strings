#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <iostream>
#include <math.h>

// EXTERNAL MODULES ON THE SCRIPTS FOLDER 

#include "input.h"
#include "init.h"
#include "spatial.h"
#include "stringID.h"

void show_parameters();

// TODO:
void compute_phi1_kernel(float *kernel, float *phi1, float *phi2, float * phi1dot);
void compute_phi2_kernel(float *kernel, float *phi1, float *phi2, float * phi2dot);
void copy_array(float * src, float * dest, int length);
void phi1_drift(float * phi1, float * phi1dot);
void phi2_drift(float * phi2, float * phi2dot);
void phi1_kick(float * phi1dot, float * K1, float * K1_next);
void phi2_kick(float * phi2dot, float * K2, float * K2_next);
void compute_axion(float * axion, float * phi1, float * phi2);
void compute_saxion(float * saxion, float * phi1, float * phi2);

int main(int argc, char *argv[]) {

    // TODO: allocate memory
    // Can have array of arrays: phi1[N][N] (for 2D case)
    //                           phi1[N][N][N] (for 3D case)
    // or a single array with length N*NDIMS: phi1[N*NDIMS] (works for both 2D and 3D cases)
    float *phi1;
    float *phi2;
    float *phi1dot;
    float *phi2dot;
    float *axion;
    float *saxion;

    // INITIAL CONDITIONS
    // init_noise(phi1, phi2, phi1dot, phi2dot);
    gaussian_thermal(phi1, phi2, phi1dot, phi2dot, false);

    // INTIALIZE KICKS K1, K2

    // TODO: allocate memory
    float *K1;
    float *K2;
    float *K1_next;
    float *K2_next;

    compute_phi1_kernel(K1_next, phi1, phi2, phi1dot); // K1_next = Laplacian(phi1,dx,N) - 2*(Era/t_evol)*phidot1 - lambdaPRS*phi1*(phi1**2.0+phi2**2.0 - 1 + (T0/L)**2.0/(3.0*t_evol**2.0))
    compute_phi2_kernel(K2_next, phi1, phi2, phi2dot); // K2_next = Laplacian(phi2,dx,N) - 2*(Era/t_evol)*phidot2 - lambdaPRS*phi2*(phi1**2.0+phi2**2.0 - 1 + (T0/L)**2.0/(3.0*t_evol**2.0))

    copy_array(K1_next, K1, N * NDIMS); // K1 = 1.0*K1_next// K1 = 1.0*K1_next
    copy_array(K2_next, K2, N * NDIMS); // K2 = 1.0*K2_next// K2 = 1.0*K2_next

    int final_step = light_time - round(1 / DeltaRatio) + 1;
    // z_slice = randrange(N-1) # FOR 3D SLICE PLOT 

    for (int tstep = 0; tstep < final_step; tstep++) {

        // TIME EVOLUTON ALGORITHM 
   
        phi1_drift(phi1, phi1dot); // phi1 = phi1 + dtau*(phidot1 + 0.5*K1*dtau)
        phi2_drift(phi2, phi2dot); // phi2 = phi2 + dtau*(phidot2 + 0.5*K2*dtau)

        t_evol = t_evol + dtau;

        compute_phi1_kernel(K1_next, phi1, phi2, phi1dot); // K1_next = Laplacian(phi1,dx,N) - 2*(Era/t_evol)*phidot1 - lambdaPRS*phi1*(phi1**2.0+phi2**2.0 - 1 + (T0/L)**2.0/(3.0*t_evol**2.0))
        compute_phi2_kernel(K2_next, phi1, phi2, phi2dot); // K2_next = Laplacian(phi2,dx,N) - 2*(Era/t_evol)*phidot2 - lambdaPRS*phi2*(phi1**2.0+phi2**2.0 - 1 + (T0/L)**2.0/(3.0*t_evol**2.0))

        phi1_kick(phi1dot, K1, K1_next); // phidot1 = phidot1 + 0.5*(K1 + K1_next)*dtau
        phi2_kick(phi2dot, K2, K2_next); // phidot2 = phidot2 + 0.5*(K2 + K2_next)*dtau

        copy_array(K1_next, K1, N * NDIMS); // K1 = 1.0*K1_next
        copy_array(K2_next, K2, N * NDIMS); // K2 = 1.0*K2_next

        // TIME VARIABLES 

        float kappa = log(t_evol/ms); // String tension (also time variable)
        float R = t_evol/(ms*L); // Scale factor in L units
        float time = t0 * powf(R/R0*ms, 2.0f); // Cosmic time in L units 

        // PHYSICAL FIELDS
        // TODO: complex numbers
        // PHI = phi1 + 1j * phi2
        // PHIDOT = phidot1 +1j * phidot2
        // axiondot = (PHIDOT/PHI).imag
        // adot_screen = saxion*axiondot
        compute_axion(axion, phi1, phi2); // axion = arctan2(phi1,phi2) 
        compute_saxion(saxion, phi1, phi2); // saxion = sqrt(phi1**2.0+phi2**2.0) 

        // STRINGS  
    
        // Nchecks = 100
        // to_analyse = arange(0, final_step,int(final_step/Nchecks))
        // thr = 1 # In % value
        // if tstep in to_analyse:
        //     num_cores = cores_pi(axion,N,thr)
        //     xi_2D = num_cores*time**2.0/(t_evol**2.0)
        //     mu = pi*kappa
        //     string_en = mu*xi_2D/time**2.0
    }

    // PLOT FINAL OUTPUT

    //fig_axion,ax_axion = TwilightPlot(axion,t_evol)
    //Snap_Save(fig_axion,tstep,mov_dir2D)

    return EXIT_SUCCESS;
}

void show_parameters() {

    printf("Parameters:\n");
    std::cout << " " << "NDIMS        " << NDIMS << '\n';
    std::cout << " " << "N            " << N << '\n';
    std::cout << " " << "Delta        " << Delta << '\n';
    std::cout << " " << "DeltaRatio   " << DeltaRatio << '\n';
    std::cout << " " << "StencilOrder " << StencilOrder << '\n';
    std::cout << std::scientific;
    std::cout << " " << "fa_phys      " << fa_phys << '\n';
    std::cout << std::fixed;
    std::cout << " " << "lambdaPRS    " << lambdaPRS << '\n';
    std::cout << " " << "Potential    " << Potential << '\n';
    std::cout << " " << "Era          " << Era << '\n';
    std::cout << " " << "Mpl          " << Mpl << '\n';
    std::cout << " " << "fa           " << fa << '\n';
    std::cout << " " << "ms           " << ms << '\n';
    std::cout << " " << "L            " << L << '\n';
    std::cout << " " << "Delta_tau    " << Delta_tau << '\n';
    std::cout << " " << "H1           " << H1 << '\n';
    std::cout << " " << "dx           " << dx << '\n';
    std::cout << " " << "dtau         " << dtau << '\n';
    std::cout << " " << "t_evol       " << t_evol << '\n';
    std::cout << " " << "light_time   " << light_time << '\n';
    std::cout << " " << "gstar        " << gstar << '\n';
    std::cout << " " << "T0           " << T0 << '\n';
    std::cout << " " << "R0           " << R0 << '\n';
    std::cout << " " << "t0           " << t0 << '\n';
    std::cout << " " << "meffsquared  " << meffsquared << '\n';

}