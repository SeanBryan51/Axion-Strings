/*
 * Description: Module that sets the initial conditions for the string simulation
 */

#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_math.h>
// #include <fftw3.h>

#include "../parameters.h"
#include "init.h"
#include "spatial.h"

/*
 * Random white noise in position space, independent of the shape
 * of the potential.
 */
void init_noise(float *phi1, float *phi2, float *phidot1, float *phidot2) {

    // TODO: Gadget2 generates a seed table from the single seed value, is this better?

    gsl_rng *rng = gsl_rng_alloc(gsl_rng_ranlxs0);
    gsl_rng_set(rng, globals.seed);

    float th, r;
    if (globals.NDIMS == 2) {
        for (int i = 0; i < globals.N; i++) {
            for (int j = 0; j < globals.N; j++) {
                th = 2 * M_PI * gsl_rng_uniform(rng);
                r = gsl_ran_gaussian(rng, 0.1f) + 1.0f;
                // Note: offset(x,y) = (x + ny * y)
                phi1[offset2(i,j,globals.N)] = r * cosf(th);
                phi2[offset2(i,j,globals.N)] = r * sinf(th);
                phidot1[offset2(i,j,globals.N)] = phidot2[offset2(i,j,globals.N)] = 0;
            }
        }
    } else if (globals.NDIMS == 3) {
        for (int i = 0; i < globals.N; i++) {
            for (int j = 0; j < globals.N; j++) {
                for (int k = 0; k < globals.N; k++) {
                    th = 2 * M_PI * gsl_rng_uniform(rng);
                    r = gsl_ran_gaussian(rng, 0.1f) + 1.0f;
                    // Note: offset(x,y,z) = (x * ny + y) * nz + z
                    phi1[offset3(i,j,j,globals.N)] = r * cosf(th);
                    phi2[offset3(i,j,j,globals.N)] = r * sinf(th);
                    phidot1[offset3(i,j,j,globals.N)] = phidot2[offset3(i,j,j,globals.N)] = 0;
                }
            }
        }
    }
}

void gaussian_thermal(float * phi1, float * phi2, float * phidot1, float *phidot2, int flag_normalise) {
    // TODO

}

// def fftind(N):
//     k_ind = np.mgrid[:N, :N] - int( (N + 1)/2 )
//     k_ind = scipy.fftpack.fftshift(k_ind)
//     return(k_ind)


// def IC_Thermal(L,k_scale,meffsquared,T0,N,flag_normalize = True):
    
//     k_idx = fftind(N)

//     k = np.sqrt(k_idx[0]**2 + k_idx[1]**2+1e-10)
//     omegak = np.sqrt((k*np.pi/N)**2 + meffsquared)
//     bose = 1/(np.exp(omegak/T0)-1)
//     amplitude = np.sqrt(L*bose/omegak) # Power spectrum for phi
//     amplitude_dot = np.sqrt(L*bose*omegak) # Power spectrum for phidot
    
//     if NDIMS == 2:
        
//         noise = random.normal(size = (N, N)) + 1j*random.normal(size = (N,N)) 
        
//         if single_precision:
            
//             noise = noise.astype('float32')
        
//         gfield1 = scipy.fft.ifft2(noise*amplitude).real
//         gfield2 = scipy.fft.ifft2(noise*amplitude).imag

//         if flag_normalize:

//             gfield1 = gfield1 - np.mean(gfield1)
//             gfield1 = gfield1/np.std(gfield1)

//         if flag_normalize:

//             gfield2 = gfield2 - np.mean(gfield2)
//             gfield2 = gfield2/np.std(gfield2)
          
//         gfield1_dot = scipy.fft.ifft2(noise*amplitude_dot).real
//         gfield2_dot = scipy.fft.ifft2(noise*amplitude_dot).imag

//         if flag_normalize:
            
//             gfield1_dot = gfield1_dot - np.mean(gfield1_dot)
//             gfield1_dot = gfield1_dot/np.std(gfield1_dot)
        
//         if flag_normalize:
            
//             gfield2_dot = gfield2_dot - np.mean(gfield2_dot)
//             gfield2_dot = gfield2_dot/np.std(gfield2_dot)
    
//     if NDIMS == 3:
        
//         noise = random.normal(size = (N,N,N)) + 1j*random.normal(size = (N,N,N))
        
//         if single_precision:
            
//             noise = noise.astype('float32')
            
//         gfield1 = scipy.fft.ifftn(noise*amplitude).real
//         gfield2 = scipy.fft.ifftn(noise*amplitude).imag

//         if flag_normalize:

//             gfield1 = gfield1 - np.mean(gfield1)
//             gfield1 = gfield1/np.std(gfield1)

//         if flag_normalize:

//             gfield2 = gfield2 - np.mean(gfield2)
//             gfield2 = gfield2/np.std(gfield2)

           
//         gfield1_dot = scipy.fft.ifftn(noise*amplitude_dot).real
//         gfield2_dot = scipy.fft.ifftn(noise*amplitude_dot).imag

//         if flag_normalize:
            
//             gfield1_dot = gfield1_dot - np.mean(gfield1_dot)
//             gfield1_dot = gfield1_dot/np.std(gfield1_dot)
        
//         if flag_normalize:
            
//             gfield2_dot = gfield2_dot - np.mean(gfield2_dot)
//             gfield2_dot = gfield2_dot/np.std(gfield2_dot)

//     return gfield1,gfield2,gfield1_dot,gfield2_dot
