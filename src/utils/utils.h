#pragma once

#include "../standard/common.h"

void read_ic_file(dtype * phi1, dtype * phidot1, dtype * phi2, dtype * phidot2, const char * filepath);
void save_data(char *file_name, dtype *data, int length);
void initialise_globals(char *parameter_file);
void read_parameter_file(char *fname);