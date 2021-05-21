
#include <iostream> 
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <vector>

using namespace std;

// External modules 

#include "stringID.h"


// 2D version

int Cores2D(){ // To insert arguments of function 
    
    // Initialize list to push back string locations  
	vector <double> s;
	double accept = 0.5 - 0.5*thr/100;
	int count = 0;

	for (int=0,i<N;i++){
		for (int j=0;j<N;j++){

            	norm1 = (field[i][j] + M_PI)/(2*M_PI)
            	norm2 = (field[i+1][j] + M_PI)/(2*M_PI)
            	norm3 = (field[i+1][j+1] + M_PI)/(2*M_PI)
            	norm4 = (field[i][j+1] + M_PI)/(2*M_PI)

            	theta1 = min(abs(norm2-norm1),1-abs(norm2-norm1))
            	theta2 = min(abs(norm3-norm2),1-abs(norm3-norm2))
            	theta3 = min(abs(norm4-norm3),1-abs(norm4-norm3))
            	theta_sum = theta1 + theta2 + theta3

            	if (theta_sum > accept){
                	s.push_back([i,j])
            		}   
        	}
    	}

    	for(int a=1,a<=s.size(),a++){
		diff_y = s[a+1][1]-s[a][1]
        	diff_x = s[a+1][0]-s[a][0]

        	if(diff_y == 0 and diff_x == 1){
            		count+=1
			}
        
        	if(diff_y == 1 and diff_x == 0){
            		count+=1
			}
    	}
    return s.size()-count
}


// 3D version

int Cores3D(){ // To insert arguments of function 
    
    vector <double> s;
    double accept = 0.5 - 0.5*thr/100;
    int count = 0;

    
    for (int=0,i<N;i++){
        for (int j=0;j<N;j++){
            for (int k=0;k<N;k++){

                norm1a = (field[i][j][k] + M_PI)/(2*M_PI)
                norm2a = (field[i+1][j][k] + M_PI)/(2*M_PI)
                norm3a = (field[i+1][j+1][k] + M_PI)/(2*M_PI)
                norm4a = (field[i][j+1][k] + M_PI)/(2*M_PI)

                norm1b = (field[i][j][k] + M_PI)/(2*M_PI)
                norm2b = (field[i+1][j][k] + M_PI)/(2*M_PI)
                norm3b = (field[i][j+1][k+1] + M_PI)/(2*M_PI)
                norm4b = (field[i][j][k+1] + M_PI)/(2*M_PI)

                norm1c = (field[i][j][k]+M_PI)/(2*M_PI)         
                norm2c = (field[i][j+1][k]+M_PI)/(2*M_PI)
                norm3c = (field[i][j+1][k+1]+M_PI)/(2*M_PI)
                norm4c = (field[i][j][k+1]+M_PI)/(2*M_PI)

                theta1a = min(abs(norm2a-norm1a),1-abs(norm2a-norm1a))
                theta2a = min(abs(norm3a-norm2a),1-abs(norm3a-norm2a))
                theta3a = min(abs(norm4a-norm3a),1-abs(norm4a-norm3a))

                theta1b = min(abs(norm2b-norm1b),1-abs(norm2b-norm1b))
                theta2b = min(abs(norm3b-norm2b),1-abs(norm3b-norm2b))
                theta3b = min(abs(norm4b-norm3b),1-abs(norm4b-norm3b))
            
                theta1c = min(abs(norm2c-norm1c),1-abs(norm2c-norm1c))
                theta2c = min(abs(norm3c-norm2c),1-abs(norm3c-norm2c))
                theta3c = min(abs(norm4c-norm3c),1-abs(norm4c-norm3c))

                theta_sum_a = theta1a + theta2a + theta3a
                theta_sum_b = theta1b + theta2b + theta3b
                theta_sum_c = theta1c + theta2c + theta3c
                
                theta_sum_a = theta1a + theta2a + theta3a
                theta_sum_b = theta1b + theta2b + theta3b
                theta_sum_c = theta1c + theta2c + theta3c

                if (theta_sum_a || theta_sum_b || theta_sum_c > accept){
                    s.push_back([i,j,k])
                }
            }
        }
    }

    for(int a=0,a<s.size(),a++){

        diff_y = s[a+1][1]-s[a][1]
        diff_x = s[a+1][0]-s[a][0]

        if(diff_y == 0 and diff_x == 1){
            count+=1
        }
        
        if(diff_y == 1 and diff_x == 0){
            count+=1
        }
    }


    return s.size()-count
}


