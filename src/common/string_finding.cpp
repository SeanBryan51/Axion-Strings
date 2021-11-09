#include "common.hpp"

int cores2(data_t *field, std::vector <vec2i> &list) {

    int N = parameters.N;
    int length = get_length();
    int thr = parameters.thr;
    data_t accept = 0.5f * (1.0f - thr/100.0f);

    #pragma omp declare reduction (merge : std::vector<vec2i> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))

    #pragma omp parallel for reduction(merge: list)
    for (int m = 0; m < length; m++) {
        int i, j;
        coordinate2(&i, &j, m, N, 0);

        data_t norm1 = (field[offset2(i,j,N,0)] + M_PI)/(2*M_PI);
        data_t norm2 = (field[offset2(i+1,j,N,0)] + M_PI)/(2*M_PI);
        data_t norm3 = (field[offset2(i+1,j+1,N,0)] + M_PI)/(2*M_PI);
        data_t norm4 = (field[offset2(i,j+1,N,0)] + M_PI)/(2*M_PI);

        data_t theta1 = std::min(abs(norm2-norm1),1-abs(norm2-norm1));
        data_t theta2 = std::min(abs(norm3-norm2),1-abs(norm3-norm2));
        data_t theta3 = std::min(abs(norm4-norm3),1-abs(norm4-norm3));
        data_t theta_sum = theta1 + theta2 + theta3;

        if (theta_sum > accept) {
            list.push_back((vec2i) {i, j});
        }
    }

    return list.size();
}

int cores3(data_t *field, std::vector <vec3i> &list) {

    int N = parameters.N;
    int length = get_length();
    int thr = parameters.thr;
    data_t accept = 0.5f * (1.0f - thr/100.0f);

    #pragma omp declare reduction (merge : std::vector<vec3i> : omp_out.insert(omp_out.end(), omp_in.begin(), omp_in.end()))

    #pragma omp parallel for reduction(merge: list)
    for (int m = 0; m < length; m++) {
        int i, j, k;
        coordinate3(&i, &j, &k, m, N);

        data_t norm1a = (field[offset3(i,j,k,N)] + M_PI) / (2.0f*M_PI);
        data_t norm2a = (field[offset3(i+1,j,k,N)] + M_PI) / (2.0f*M_PI);
        data_t norm3a = (field[offset3(i+1,j+1,k,N)] + M_PI) / (2.0f*M_PI);
        data_t norm4a = (field[offset3(i,j+1,k,N)] + M_PI) / (2.0f*M_PI);

        data_t norm1b = (field[offset3(i,j,k,N)] + M_PI) / (2.0f*M_PI);
        data_t norm2b = (field[offset3(i+1,j,k,N)] + M_PI) / (2.0f*M_PI);
        data_t norm3b = (field[offset3(i,j+1,k+1,N)] + M_PI) / (2.0f*M_PI);
        data_t norm4b = (field[offset3(i,j,k+1,N)] + M_PI) / (2.0f*M_PI);

        data_t norm1c = (field[offset3(i,j,k,N)] + M_PI) / (2.0f*M_PI);
        data_t norm2c = (field[offset3(i,j+1,k,N)] + M_PI) / (2.0f*M_PI);
        data_t norm3c = (field[offset3(i,j+1,k+1,N)] + M_PI) / (2.0f*M_PI);
        data_t norm4c = (field[offset3(i,j,k+1,N)] + M_PI) / (2.0f*M_PI);

        data_t theta1a = std::min(abs(norm2a-norm1a),1-abs(norm2a-norm1a));
        data_t theta2a = std::min(abs(norm3a-norm2a),1-abs(norm3a-norm2a));
        data_t theta3a = std::min(abs(norm4a-norm3a),1-abs(norm4a-norm3a));

        data_t theta1b = std::min(abs(norm2b-norm1b),1-abs(norm2b-norm1b));
        data_t theta2b = std::min(abs(norm3b-norm2b),1-abs(norm3b-norm2b));
        data_t theta3b = std::min(abs(norm4b-norm3b),1-abs(norm4b-norm3b));

        data_t theta1c = std::min(abs(norm2c-norm1c),1-abs(norm2c-norm1c));
        data_t theta2c = std::min(abs(norm3c-norm2c),1-abs(norm3c-norm2c));
        data_t theta3c = std::min(abs(norm4c-norm3c),1-abs(norm4c-norm3c));

        data_t theta_sum_a = theta1a + theta2a + theta3a;
        data_t theta_sum_b = theta1b + theta2b + theta3b;
        data_t theta_sum_c = theta1c + theta2c + theta3c;

        if (theta_sum_a > accept || theta_sum_b > accept || theta_sum_c > accept) {
            list.push_back((vec3i) {i,j,k});
        }
    }

    return list.size();
}
