
#include <iostream> 
// #include <cstdlib>
#include <gsl/gsl_math.h>
#include <algorithm>
#include <vector>

using namespace std;

// External modules 

#include "common.h"
#include "../parameters.h"

typedef struct vec2i {
    int x;
    int y;
} vec2i;

typedef struct vec3i {
    int x;
    int y;
    int z;
} vec3i;

// 2D version

int Cores2D(dtype *field, int thr) {
    
    // Initialize list to push back string locations  
    vector <vec2i> s;

    dtype accept = 0.5f - 0.5f*thr/100;
    int count = 0;
    int N = parameters.N;

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++){

            dtype norm1 = (field[offset2(i,j,N)] + M_PI)/(2*M_PI);
            dtype norm2 = (field[offset2(i+1,j,N)] + M_PI)/(2*M_PI);
            dtype norm3 = (field[offset2(i+1,j+1,N)] + M_PI)/(2*M_PI);
            dtype norm4 = (field[offset2(i,j+1,N)] + M_PI)/(2*M_PI);

            dtype theta1 = min(abs(norm2-norm1),1-abs(norm2-norm1));
            dtype theta2 = min(abs(norm3-norm2),1-abs(norm3-norm2));
            dtype theta3 = min(abs(norm4-norm3),1-abs(norm4-norm3));
            dtype theta_sum = theta1 + theta2 + theta3;

            if (theta_sum > accept) {
                s.push_back((vec2i) {i, j});
            }
        }
    }

    for (int a = 0; a < s.size() - 1; a++) {
        int diff_y = s[a+1].y - s[a].y;
        int diff_x = s[a+1].x - s[a].x;

        if (diff_y == 0 && diff_x == 1) {
            count += 1;
        }
        
        if (diff_y == 1 && diff_x == 0) {
            count += 1;
        }
    }

    return s.size() - count;
}


// 3D version

int Cores3D(dtype *field, int thr) {
    
    vector <vec3i> s;
    dtype accept = 0.5f - 0.5f*thr/100;
    int count = 0;
    int N = parameters.N;
    
    for (int i = 0; i < N; i++) {
        for (int j=0; j < N; j++) {
            for (int k = 0; k < N; k++) {

                dtype norm1a = (field[offset3(i,j,k,N)] + M_PI)/(2*M_PI);
                dtype norm2a = (field[offset3(i+1,j,k,N)] + M_PI)/(2*M_PI);
                dtype norm3a = (field[offset3(i+1,j+1,k,N)] + M_PI)/(2*M_PI);
                dtype norm4a = (field[offset3(i,j+1,k,N)] + M_PI)/(2*M_PI);

                dtype norm1b = (field[offset3(i,j,k,N)] + M_PI)/(2*M_PI);
                dtype norm2b = (field[offset3(i+1,j,k,N)] + M_PI)/(2*M_PI);
                dtype norm3b = (field[offset3(i,j+1,k+1,N)] + M_PI)/(2*M_PI);
                dtype norm4b = (field[offset3(i,j,k+1,N)] + M_PI)/(2*M_PI);

                dtype norm1c = (field[offset3(i,j,k,N)]+M_PI)/(2*M_PI);
                dtype norm2c = (field[offset3(i,j+1,k,N)]+M_PI)/(2*M_PI);
                dtype norm3c = (field[offset3(i,j+1,k+1,N)]+M_PI)/(2*M_PI);
                dtype norm4c = (field[offset3(i,j,k+1,N)]+M_PI)/(2*M_PI);

                dtype theta1a = min(abs(norm2a-norm1a),1-abs(norm2a-norm1a));
                dtype theta2a = min(abs(norm3a-norm2a),1-abs(norm3a-norm2a));
                dtype theta3a = min(abs(norm4a-norm3a),1-abs(norm4a-norm3a));

                dtype theta1b = min(abs(norm2b-norm1b),1-abs(norm2b-norm1b));
                dtype theta2b = min(abs(norm3b-norm2b),1-abs(norm3b-norm2b));
                dtype theta3b = min(abs(norm4b-norm3b),1-abs(norm4b-norm3b));
            
                dtype theta1c = min(abs(norm2c-norm1c),1-abs(norm2c-norm1c));
                dtype theta2c = min(abs(norm3c-norm2c),1-abs(norm3c-norm2c));
                dtype theta3c = min(abs(norm4c-norm3c),1-abs(norm4c-norm3c));

                dtype theta_sum_a = theta1a + theta2a + theta3a;
                dtype theta_sum_b = theta1b + theta2b + theta3b;
                dtype theta_sum_c = theta1c + theta2c + theta3c;
                
                if (theta_sum_a || theta_sum_b || theta_sum_c > accept){
                    s.push_back((vec3i) {i,j,k});
                }
            }
        }
    }

    for (int a = 0; a < s.size() - 1; a++) {

        int diff_y = s[a+1].y - s[a].y;
        int diff_x = s[a+1].x - s[a].x;

        if (diff_y == 0 && diff_x == 1) {
            count += 1;
        }
        
        if (diff_y == 1 && diff_x == 0) {
            count += 1;
        }
    }

    return s.size() - count;
}


