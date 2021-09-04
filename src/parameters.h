#pragma once

#define MAX_LEN 200

extern struct _parameters {

    // User defined parameters to be read from 
    // parameter file:

    int   NDIMS;
    int   N;
    float space_step;
    float time_step;

    float fa_phys;
    float lambdaPRS;
    char  potential[MAX_LEN];
    char  time_variable[MAX_LEN];

    int   save_snapshots;
    int   n_snapshots;
    char  output_directory[MAX_LEN];

    int   write_output_file;
    char  output_file_path[MAX_LEN];

    unsigned int seed;

    // String finding:
    int run_string_finding;
    int string_checks;
    int thr;
    char string_finding_output_file_path[MAX_LEN];

} parameters;
