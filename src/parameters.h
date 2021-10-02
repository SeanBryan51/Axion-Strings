#pragma once

#define MAX_LEN 200

extern struct _parameters {

    // User defined parameters to be read from 
    // parameter file:

    float lambdaPRS;
    int   NDIMS;
    int   N;
    float space_step;
    float time_step;
    int   stencil_setting;
    unsigned int seed;

    int  write_output_file;
    char output_file_path[MAX_LEN];

    int  save_snapshots;
    int  n_snapshots;
    char output_directory[MAX_LEN];
    int  save_fields;
    int  save_strings;

    int  sample_time_series;
    int  n_samples;
    char ts_output_path[MAX_LEN];

    int sample_strings;
    int sample_background;

    int thr;

} parameters;
