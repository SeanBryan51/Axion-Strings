#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#include "standard/interface.h"
#include "utils/utils.h"

int main(int argc, char *argv[]) {

    if(argc != 2) {
        printf("Error: usage ./main <ParameterFile>\n");
        return EXIT_FAILURE;
    }

    // Initialise global variables defined in parameters.h:
    read_parameter_file(argv[1]);

    // TODO: parameter sanity checks
    printf("Running with %s\n", argv[1]);

    double start = omp_get_wtime();

    run_standard();

    printf("Time taken: %f seconds\n", omp_get_wtime() - start);

    return EXIT_SUCCESS;
}