#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "standard/interface.h"
#include "utils/utils.h"

int main(int argc, char *argv[]) {

    if(argc != 2) {
        printf("Error: usage ./main <ParameterFile>\n");
        return EXIT_FAILURE;
    }

    // Set global variables from parameter file:
    initialise_globals(argv[1]);

    clock_t start, end;
    double cpu_time_used;

    start = clock();

    run_standard();

    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

    printf("Time taken: %f seconds\n", cpu_time_used);

    return EXIT_SUCCESS;
}