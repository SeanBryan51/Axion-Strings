#include "s_internal.hpp"


void read_field_data(const char *filepath, data_t *data, int length) {

    FILE *fp = fopen(filepath, "r");
    assert(fp != NULL);

    fread(data, sizeof(data_t), length, fp);

    fclose(fp);
}

void save_data(char *file_name, data_t *data, int length) {

    char *path = (char *) alloca(sizeof(parameters.output_directory) + sizeof(file_name) + 1);
    assert(path != NULL);
    sprintf(path, "%s/%s", parameters.output_directory, file_name);

    fprintf(fp_main_output, "  Saving data... at %s\n", path);

    FILE *fp = fopen(path, "w");
    assert(fp != NULL);

    fwrite(data, sizeof(data_t), length, fp);

    fclose(fp);
}