
#include "utils.h"
#include "mmio.h"

/* 
*   Matrix Market I/O example program
*
*   Read a real (non-complex) sparse matrix from a Matrix Market (v. 2.0) file.
*   and copies it to stdout.  This porgram does nothing useful, but
*   illustrates common usage of the Matrix Matrix I/O routines.
*   (See http://math.nist.gov/MatrixMarket for details.)
*
*   Usage:  a.out [filename] > output
*
*       
*   NOTES:
*
*   1) Matrix Market files are always 1-based, i.e. the index of the first
*      element of a matrix is (1,1), not (0,0) as in C.  ADJUST THESE
*      OFFSETS ACCORDINGLY offsets accordingly when reading and writing 
*      to files.
*
*   2) ANSI C requires one to use the "l" format modifier when reading
*      double precision floating point numbers in scanf() and
*      its variants.  For example, use "%lf", "%lg", or "%le"
*      when reading doubles, otherwise errors will occur.
*/

void read_mtx_file(char *file_name, int *ret_N, int *ret_nnz, int *rows, int *cols, dtype *values) {

    int ret_code;
    MM_typecode matcode;
    FILE *f;
    int M, N, nnz;   

    if ((f = fopen(file_name, "r")) == NULL) 
        exit(EXIT_FAILURE);

    if (mm_read_banner(f, &matcode) != 0)
    {
        printf("Could not process Matrix Market banner.\n");
        exit(EXIT_FAILURE);
    }

    /*  This is how one can screen matrix types if their application */
    /*  only supports a subset of the Matrix Market data types.      */

    if (!mm_is_matrix(matcode))
    {
        printf("Sorry, this application does not support ");
        printf("Matrix Market type: [%s]\n", mm_typecode_to_str(matcode));
        exit(EXIT_FAILURE);
    }

    /* find out size of sparse matrix .... */

    if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nnz)) !=0 && M == N)
        exit(EXIT_FAILURE);

    /* reseve memory for matrices */

    rows = (int *) malloc(nnz * sizeof(int));
    cols = (int *) malloc(nnz * sizeof(int));
    values = (dtype *) malloc(nnz * sizeof(dtype));


    /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
    /*   specifier as in "%lg", "%lf", "%le", otherwise errors will occur */
    /*  (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)            */

    for (int i = 0; i < nnz; i++)
    {
        fscanf(f, "%d %d %lg\n", &rows[i], &cols[i], &values[i]);
        rows[i]--;  /* adjust from 1-based to 0-based */
        cols[i]--;
    }

    if (f !=stdin) fclose(f);

    *ret_N = N;
    *ret_nnz = nnz;
}
