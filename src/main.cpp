#include <stdio.h>
#include <stdlib.h>

#include "standard/interface.h"
#include "utils/utils.h"

int main(int argc, char *argv[]) {

    if(argc != 2) {
        printf("Error: usage ./main <ParameterFile>\n");
        return EXIT_FAILURE;
    }

    // Set global variables from parameter file:
    initialise_globals(argv[1]);

    run_standard();

    return EXIT_SUCCESS;
}