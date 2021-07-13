#pragma once

#include "../standard/common.h"

void read_ic_file(dtype * phi1, dtype * phidot1, dtype * phi2, dtype * phidot2, const char * filepath);
void read_mtx_file(char *file_name, int *N, int *nnz, int *rows, int *cols, dtype *values);
void save_data(char *file_name, dtype *data, int length);
void read_parameter_file(char *fname);