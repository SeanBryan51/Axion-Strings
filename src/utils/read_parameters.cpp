#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "utils.h"
#include "../parameters.h"

#define MAX_TAGS 300
#define MAX_LINE_WIDTH 200

struct _parameters parameters;

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
static int parameters_to_read(Parameter *p_list) {

    int n_params = 0;

    // Add additional parameters here:

    strcpy(p_list[n_params].tag, "lambdaPRS");
    p_list[n_params].addr = &parameters.lambdaPRS;
    p_list[n_params].type = FLOAT;
    n_params++;

    strcpy(p_list[n_params].tag, "NDIMS");
    p_list[n_params].addr = &parameters.NDIMS;
    p_list[n_params].type = INT;
    n_params++;

    strcpy(p_list[n_params].tag, "N");
    p_list[n_params].addr = &parameters.N;
    p_list[n_params].type = INT;
    n_params++;

    strcpy(p_list[n_params].tag, "space_step");
    p_list[n_params].addr = &parameters.space_step;
    p_list[n_params].type = FLOAT;
    n_params++;

    strcpy(p_list[n_params].tag, "time_step");
    p_list[n_params].addr = &parameters.time_step;
    p_list[n_params].type = FLOAT;
    n_params++;

    strcpy(p_list[n_params].tag, "seed");
    p_list[n_params].addr = &parameters.seed;
    p_list[n_params].type = INT;
    n_params++;

    strcpy(p_list[n_params].tag, "write_output_file");
    p_list[n_params].addr = &parameters.write_output_file;
    p_list[n_params].type = INT;
    n_params++;

    strcpy(p_list[n_params].tag, "output_file_path");
    p_list[n_params].addr = &parameters.output_file_path;
    p_list[n_params].type = STRING;
    n_params++;

    strcpy(p_list[n_params].tag, "save_snapshots");
    p_list[n_params].addr = &parameters.save_snapshots;
    p_list[n_params].type = INT;
    n_params++;

    strcpy(p_list[n_params].tag, "n_snapshots");
    p_list[n_params].addr = &parameters.n_snapshots;
    p_list[n_params].type = INT;
    n_params++;

    strcpy(p_list[n_params].tag, "output_directory");
    p_list[n_params].addr = &parameters.output_directory;
    p_list[n_params].type = STRING;
    n_params++;

    strcpy(p_list[n_params].tag, "save_fields");
    p_list[n_params].addr = &parameters.save_fields;
    p_list[n_params].type = INT;
    n_params++;

    strcpy(p_list[n_params].tag, "save_strings");
    p_list[n_params].addr = &parameters.save_strings;
    p_list[n_params].type = INT;
    n_params++;

    strcpy(p_list[n_params].tag, "sample_time_series");
    p_list[n_params].addr = &parameters.sample_time_series;
    p_list[n_params].type = INT;
    n_params++;

    strcpy(p_list[n_params].tag, "n_samples");
    p_list[n_params].addr = &parameters.n_samples;
    p_list[n_params].type = INT;
    n_params++;

    strcpy(p_list[n_params].tag, "ts_output_path");
    p_list[n_params].addr = &parameters.ts_output_path;
    p_list[n_params].type = STRING;
    n_params++;

    strcpy(p_list[n_params].tag, "sample_strings");
    p_list[n_params].addr = &parameters.sample_strings;
    p_list[n_params].type = INT;
    n_params++;

    strcpy(p_list[n_params].tag, "sample_background");
    p_list[n_params].addr = &parameters.sample_background;
    p_list[n_params].type = INT;
    n_params++;

    strcpy(p_list[n_params].tag, "thr");
    p_list[n_params].addr = &parameters.thr;
    p_list[n_params].type = INT;
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

    Parameter p_list[MAX_TAGS];
    int n_params = parameters_to_read(p_list);

    char line[MAX_LINE_WIDTH];
    char buf1[MAX_LINE_WIDTH], buf2[MAX_LINE_WIDTH], buf3[MAX_LINE_WIDTH];

    FILE *fd = fopen(fname, "r");
    assert(fd != NULL);

    while(fgets(line, MAX_LINE_WIDTH, fd) != NULL) {
        if(sscanf(line, "%s%s%s", buf1, buf2, buf3) >= 2 && buf1[0] != '%') {

            int index = -1;
            for(int i = 0; i < n_params; i++) {
                if(strcmp(buf1, p_list[i].tag) == 0) {
                    p_list[i].tag[0] = 0; // clear parameter tag
                    index = i;
                    break;
                }
            }

            if(index != -1) {
                switch (p_list[index].type) {
                    case DOUBLE:
                        *((double *) p_list[index].addr) = atof(buf2);
                        break;
                    case STRING:
                        assert(strlen(buf2) < MAX_LEN);
                        strcpy((char *) p_list[index].addr, buf2);
                        break;
                    case FLOAT:
                        *((float *) p_list[index].addr) = (float) atof(buf2);
                        break;
                    case INT:
                        *((int *) p_list[index].addr) = atoi(buf2);
                        break;
                }
            } else {
                printf("Error in file %s:   Tag '%s' not allowed or multiple defined.\n", fname, buf1);
                exit(EXIT_FAILURE);
            }
        }
    }

    fclose(fd);

    // Check for any missing parameters:
    for(int i = 0; i < n_params; i++) {
        if(p_list[i].tag[0]) {
            printf("Error. I miss a value for tag '%s' in parameter file '%s'.\n", p_list[i].tag, fname);
            exit(EXIT_FAILURE);
        }
    }

}
