
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "parameters.h"
#include "utils.h"

/* 
 * Read initial field values from the initial conditions file. Field values/arrays are stored on the heap.
 */
void read_ic_file(float * phi1, float * phidot1, float * phi2, float * phidot2, const char * filepath) {
    // TODO
    return;
}

void save_data(char *file_name, float *data, int length) {

    char *path = (char *) alloca(sizeof(globals.output_directory) + sizeof(file_name) + 1);
    assert(path);
    sprintf(path, "%s/%s", globals.output_directory, file_name);

    printf("Saving data... at %s\n", path);

    FILE *fp = fopen(path, "w");
    assert(fp != NULL);

    fwrite(data, sizeof(float), length, fp);

    fclose(fp);
    
}

