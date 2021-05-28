
#include <stdlib.h>
#include <assert.h>
#include <string.h>

#include "utils/utils.h"
#include "parameters.h"

void test_read_parameters() {
    assert(globals.NDIMS == 2);
    assert(globals.N == 256);
    assert(globals.Delta == 1.0f);
    assert(globals.DeltaRatio == 0.3333333333f);
    assert(globals.StencilOrder == 2);
    assert(globals.fa_phys == 1.0e12f);
    assert(globals.lambdaPRS == 1.0f);
    assert(strcmp(globals.Potential, "Thermal") == 0);
    assert(globals.Era == 1.0f);
    // assert(globals.run_with_ic == 0);
    // assert(strcmp(globals.ic_file_path, "asdf.txt") == 0);
    assert(globals.save_snapshots == 1);
    assert(globals.seed == 2021);
    assert(globals.n_snapshots == 69);
    assert(strcmp(globals.output_directory, "foobar") == 0);
}

int main() {

    initialise_globals((char *) "/Users/seanbryan/Documents/UNI/2021T1/Project/Axion-Strings/src/tests/test_parameters.param");

    test_read_parameters();

    return EXIT_SUCCESS;
}
