#include "common.hpp"

FILE *fp_main_output, *fp_time_series, *fp_snapshot_timings;

void fio_read_field_data(char *file_name, data_t *data, int length) {

    char *path = (char *) alloca(sizeof(parameters.output_directory) + sizeof(file_name) + 1);
    assert(path != NULL);
    sprintf(path, "%s/%s", parameters.output_directory, file_name);

    FILE *fp = fopen(path, "r");
    assert(fp != NULL);

    fread(data, sizeof(data_t), length, fp);

    fclose(fp);
}

void fio_save_field_data(char *file_name, data_t *data, int length) {

    char *path = (char *) alloca(sizeof(parameters.output_directory) + sizeof(file_name) + 1);
    assert(path != NULL);
    sprintf(path, "%s/%s", parameters.output_directory, file_name);

    fprintf(fp_main_output, "  Saving data... at %s\n", path);

    FILE *fp = fopen(path, "w");
    assert(fp != NULL);

    fwrite(data, sizeof(data_t), length, fp);

    fclose(fp);
}

void fio_save_field_data_as_slice(char *file_name, data_t *data, int length, int N) {

    char *path = (char *) alloca(sizeof(parameters.output_directory) + sizeof(file_name) + 1);
    assert(path != NULL);
    sprintf(path, "%s/%s", parameters.output_directory, file_name);

    fprintf(fp_main_output, "  Saving data... at %s\n", path);

    FILE *fp = fopen(path, "w");
    assert(fp != NULL);

    // A bit hacky but this should save a slice from one of the faces of the 3-dimensional volume.
    fwrite(data, sizeof(data_t), N * N, fp);

    fclose(fp);
}

void fio_save_flagged_data(char *file_name, int *data, int length) {

    char *path = (char *) alloca(sizeof(parameters.output_directory) + sizeof(file_name) + 1);
    assert(path != NULL);
    sprintf(path, "%s/%s", parameters.output_directory, file_name);

    fprintf(fp_main_output, "  Saving data... at %s\n", path);

    FILE *fp = fopen(path, "w");
    assert(fp != NULL);

    fwrite(data, sizeof(int), length, fp);

    fclose(fp);
}

void fio_save_pk(char *file_name, data_t *pk, data_t *ks, int *count, int n_bins) {

    char *path = (char *) alloca(sizeof(parameters.output_directory) + sizeof(file_name) + 1);
    assert(path != NULL);
    sprintf(path, "%s/%s", parameters.output_directory, file_name);

    fprintf(fp_main_output, "  Saving data... at %s\n", path);

    FILE *fp = fopen(path, "w");
    assert(fp != NULL);

    fprintf(fp, "k,pk,n_modes\n");
    for (int i = 0; i < n_bins; i++) {
        if(count[i]) fprintf(fp, "%e,%e,%d\n", ks[i], pk[i], count[i]);
    }

    fclose(fp);
}

void fio_save_strings2(char *file_name, std::vector <vec2i> *v) {

    char *path = (char *) alloca(sizeof(parameters.output_directory) + sizeof(file_name) + 1);
    assert(path != NULL);
    sprintf(path, "%s/%s", parameters.output_directory, file_name);

    fprintf(fp_main_output, "  Saving string positions... at %s\n", path);

    FILE *fp = fopen(path, "w");
    assert(fp != NULL);

    for (std::vector<vec2i>::iterator it = v->begin() ; it != v->end(); ++it) fprintf(fp, "%d, %d\n", it->x, it->y);

    fclose(fp);
}

void fio_save_strings3(char *file_name, std::vector <vec3i> *v) {

    char *path = (char *) alloca(sizeof(parameters.output_directory) + sizeof(file_name) + 1);
    assert(path != NULL);
    sprintf(path, "%s/%s", parameters.output_directory, file_name);

    fprintf(fp_main_output, "  Saving string positions... at %s\n", path);

    FILE *fp = fopen(path, "w");
    assert(fp != NULL);

    for (std::vector<vec3i>::iterator it = v->begin() ; it != v->end(); ++it) fprintf(fp, "%d, %d, %d\n", it->x, it->y, it->z);

    fclose(fp);
}

void fio_open_output_filestreams() {

    // set main output file stream:
    if (parameters.write_output_file) {
        fp_main_output = fopen(parameters.output_file_path, "w");
        assert(fp_main_output != NULL);
    } else {
        fp_main_output = stdout;
    }

    if (parameters.save_snapshots) {
        char *path = (char *) alloca(sizeof(parameters.output_directory) + 21 + 1);
        assert(path != NULL);
        sprintf(path, "%s/snapshot-timings.csv", parameters.output_directory);
        fp_snapshot_timings = fopen(path, "w");
        assert(fp_snapshot_timings != NULL);
    }

    // open time series file stream:
    if (parameters.sample_time_series) {
        fp_time_series = fopen(parameters.ts_output_path, "w");
        assert(fp_time_series != NULL);
    }
}

void fio_close_output_filestreams() {
    if (parameters.write_output_file) fclose(fp_main_output);
    if (parameters.save_snapshots) fclose(fp_snapshot_timings);
    if (parameters.sample_time_series) fclose(fp_time_series);
}
