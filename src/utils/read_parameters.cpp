#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <gsl/gsl_math.h>

#include "util.h"
#include "../parameters.h"

#define MAX_TAGS 300
#define MAX_LINE_WIDTH 200

struct _globals globals;

typedef enum Type {
    DOUBLE,
    STRING,
    FLOAT,
    INT
} Type;

typedef struct Parameter {
    char tag[50];
    Type type;
    void *addr;
} Parameter;

/*
 * Builds a list of 'parameter' objects to be read from the user
 * supplied parameter file.
 */
static int parameters_to_read(Parameter *parameters) {

    int n_params = 0;

    // Add additional parameters here:

    strcpy(parameters[n_params].tag, "NDIMS");
    parameters[n_params].addr = &globals.NDIMS;
    parameters[n_params].type = INT;
    n_params++;

    strcpy(parameters[n_params].tag, "N");
    parameters[n_params].addr = &globals.N;
    parameters[n_params].type = INT;
    n_params++;

    strcpy(parameters[n_params].tag, "Delta");
    parameters[n_params].addr = &globals.Delta;
    parameters[n_params].type = FLOAT;
    n_params++;

    strcpy(parameters[n_params].tag, "DeltaRatio");
    parameters[n_params].addr = &globals.DeltaRatio;
    parameters[n_params].type = FLOAT;
    n_params++;

    strcpy(parameters[n_params].tag, "StencilOrder");
    parameters[n_params].addr = &globals.StencilOrder;
    parameters[n_params].type = INT;
    n_params++;

    strcpy(parameters[n_params].tag, "fa_phys");
    parameters[n_params].addr = &globals.fa_phys;
    parameters[n_params].type = FLOAT;
    n_params++;

    strcpy(parameters[n_params].tag, "lambdaPRS");
    parameters[n_params].addr = &globals.lambdaPRS;
    parameters[n_params].type = FLOAT;
    n_params++;

    strcpy(parameters[n_params].tag, "Potential");
    parameters[n_params].addr = &globals.Potential;
    parameters[n_params].type = STRING;
    n_params++;

    strcpy(parameters[n_params].tag, "Era");
    parameters[n_params].addr = &globals.Era;
    parameters[n_params].type = FLOAT;
    n_params++;

    strcpy(parameters[n_params].tag, "save_snapshots");
    parameters[n_params].addr = &globals.save_snapshots;
    parameters[n_params].type = INT;
    n_params++;

    strcpy(parameters[n_params].tag, "n_snapshots");
    parameters[n_params].addr = &globals.n_snapshots;
    parameters[n_params].type = INT;
    n_params++;

    strcpy(parameters[n_params].tag, "output_directory");
    parameters[n_params].addr = &globals.output_directory;
    parameters[n_params].type = STRING;
    n_params++;

    strcpy(parameters[n_params].tag, "seed");
    parameters[n_params].addr = &globals.seed;
    parameters[n_params].type = INT;
    n_params++;

    return n_params;
}

/*
 * (taken from Gadget2 source code)
 *  This function parses the parameterfile in a simple way.  Each paramater
 *  is defined by a keyword (`tag'), and can be either of type double, int,
 *  or character string.  The routine makes sure that each parameter
 *  appears exactly once in the parameterfile, otherwise error messages are
 *  produced that complain about the missing parameters.
 */
void read_parameter_file(char *fname) {

    Parameter parameters[MAX_TAGS];
    int n_params = parameters_to_read(parameters);

    char line[MAX_LINE_WIDTH];
    char buf1[MAX_LINE_WIDTH], buf2[MAX_LINE_WIDTH], buf3[MAX_LINE_WIDTH];

    FILE *fd = fopen(fname, "r");
    assert(fd != NULL);

    while(fgets(line, MAX_LINE_WIDTH, fd) != NULL) {
        if(sscanf(line, "%s%s%s", buf1, buf2, buf3) >= 2 && buf1[0] != '%') {

            int index = -1;
            for(int i = 0; i < n_params; i++) {
                if(strcmp(buf1, parameters[i].tag) == 0) {
                    parameters[i].tag[0] = 0; // clear parameter tag
                    index = i;
                    break;
                }
            }

            if(index != -1) {
                switch (parameters[index].type) {
                    case DOUBLE:
                        *((double *) parameters[index].addr) = atof(buf2);
                        break;
                    case STRING:
                        assert(strlen(buf2) < MAX_LEN);
                        strcpy((char *) parameters[index].addr, buf2);
                        break;
                    case FLOAT:
                        *((float *) parameters[index].addr) = (float) atof(buf2);
                        break;
                    case INT:
                        *((int *) parameters[index].addr) = atoi(buf2);
                        break;
                }
            } else {
                printf("Error in file %s:   Tag '%s' not allowed or multiple defined.\n", fname, buf1);
            }
        }
    }

    fclose(fd);

    // Check for any missing parameters:
    for(int i = 0; i < n_params; i++) {
        if(parameters[i].tag[0]) {
            printf("Error. I miss a value for tag '%s' in parameter file '%s'.\n", parameters[i].tag, fname);
        }
    }
}

/*
 * Initialises all global variables, first reading all input parameters
 * from the user supplied parameter file and then initialising all other
 * derived parameters/variables.
 */
void initialise_globals(char *parameter_file) {

    // Initialise user supplied parameters:
    read_parameter_file(parameter_file);

    // Initialise the other derived independent parameters:

    globals.Mpl = 2.4f * 1e18 / globals.fa_phys; // Normalized to fa_phys chosen 
    globals.fa = 1.0; // In fa units
    globals.ms = sqrtf(globals.lambdaPRS) * globals.fa; // Saxion mass for PRS in ADM units
    globals.L = globals.Delta * globals.N; // Comoving 
    globals.Delta_tau = globals.DeltaRatio * globals.Delta;
    globals.H1 = globals.fa / globals.Delta;

    // Code spacings
    globals.dx = globals.ms * globals.Delta;
    globals.dtau = globals.ms * globals.Delta_tau;
    globals.t_evol = globals.dtau / globals.DeltaRatio - globals.dtau; // Such that in the first loop iteration, t_evol = 1 
    globals.light_time = round(0.5f * globals.N / globals.DeltaRatio);
    globals.gstar = 106.75f;
    globals.T0 = sqrt(globals.Mpl * 90.0f / (globals.gstar * M_PI * M_PI)); // In units of fa
    globals.R0 = globals.Delta * globals.ms / (globals.ms * globals.L);
    globals.t0 = globals.Delta / (2.0f * globals.L * globals.ms * globals.ms);
    globals.meffsquared = globals.lambdaPRS * gsl_pow_2(globals.T0) / 3.0f;
}