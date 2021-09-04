#include "common.h"

FILE *fp_main_output, *fp_string_finding;

void read_field_data(const char *filepath, dtype *data, int length) {

    FILE *fp = fopen(filepath, "r");
    assert(fp != NULL);

    fread(data, sizeof(dtype), length, fp);

    fclose(fp);
}

void save_data(char *file_name, dtype *data, int length) {

    char *path = (char *) alloca(sizeof(parameters.output_directory) + sizeof(file_name) + 1);
    assert(path != NULL);
    sprintf(path, "%s/%s", parameters.output_directory, file_name);

    fprintf(fp_main_output, "  Saving data... at %s\n", path);

    FILE *fp = fopen(path, "w");
    assert(fp != NULL);

    fwrite(data, sizeof(dtype), length, fp);

    fclose(fp);
}

void save_strings2(char *file_name, std::vector <vec2i> *v) {

    char *path = (char *) alloca(sizeof(parameters.output_directory) + sizeof(file_name) + 1);
    assert(path != NULL);
    sprintf(path, "%s/%s", parameters.output_directory, file_name);

    fprintf(fp_main_output, "  Saving string positions... at %s\n", path);

    FILE *fp = fopen(path, "w");
    assert(fp != NULL);

    for (std::vector<vec2i>::iterator it = v->begin() ; it != v->end(); ++it) fprintf(fp, "%d, %d\n", it->x, it->y);

    fclose(fp);
}

void save_strings3(char *file_name, std::vector <vec3i> *v) {

    char *path = (char *) alloca(sizeof(parameters.output_directory) + sizeof(file_name) + 1);
    assert(path != NULL);
    sprintf(path, "%s/%s", parameters.output_directory, file_name);

    fprintf(fp_main_output, "  Saving string positions... at %s\n", path);

    FILE *fp = fopen(path, "w");
    assert(fp != NULL);

    for (std::vector<vec3i>::iterator it = v->begin() ; it != v->end(); ++it) fprintf(fp, "%d, %d, %d\n", it->x, it->y, it->z);

    fclose(fp);
}

void open_output_filestreams() {

    // set main output file stream:
    if (parameters.write_output_file) {
        fp_main_output = fopen(parameters.output_file_path, "w");
        assert(fp_main_output != NULL);
    } else {
        fp_main_output = stdout;
    }

    // open string finding file stream:
    if (parameters.run_string_finding) {
        fp_string_finding = fopen(parameters.string_finding_output_file_path, "w");
        assert(fp_string_finding != NULL);
    }
}

void close_output_filestreams() {
    if (parameters.write_output_file) fclose(fp_main_output);
    if (parameters.run_string_finding) fclose(fp_string_finding);
}
