#pragma once

void read_ic_file(float * phi1, float * phidot1, float * phi2, float * phidot2, const char * filepath);
void save_data(char *file_name, float *data, int length);
void initialise_globals(char *parameter_file);
void read_parameter_file(char *fname);