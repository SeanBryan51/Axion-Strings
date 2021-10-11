#include "common/common.hpp"

int main(int argc, char *argv[]) {

    if(argc != 2) {
        fprintf(stdout, "Error: usage ./main <ParameterFile>\n");
        return EXIT_FAILURE;
    }

    // read in parameters defined in common.hpp:
    read_parameter_file(argv[1]);

    fio_open_output_filestreams();

    fprintf(fp_main_output, "Running with %s\n", argv[1]);

    double start = omp_get_wtime();

#ifdef AMR_ENABLED
    run_amr();
#else
    run_standard();
#endif

    fprintf(fp_main_output, "Time taken: %f seconds\n", omp_get_wtime() - start);

    fio_close_output_filestreams();

    return EXIT_SUCCESS;
}