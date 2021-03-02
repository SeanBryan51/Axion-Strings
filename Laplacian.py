#===============================Laplacian.py==================================#
# Created by Giovanni Pierobon 2021
#==============================================================================#

from numpy import *
import matplotlib.pyplot as plt
from numba import jit

stencil=5

# SPATIAL DERIVATIVE 2D
@jit(nopython=True)
def Laplacian_2D(phi,dx,N):
    ddphi = zeros(shape=(N,N))
    if stencil == 5:
        for i in range(0,N):
            for j in range(0,N):
                ddphi[i,j] = ((-phi[mod(i+2,N),j]+16*phi[mod(i+1,N),j]-30*phi[i,j]+16*phi[i-1,j] - phi[i-2,j])\
                     + (-phi[i,mod(j+2,N)] + 16*phi[i,mod(j+1,N)]-30*phi[i,j] + 16*phi[i,j-1] - phi[i,j-2]))/(12*dx**2.0) 
    if stencil == 7:
        for i in range(0,N):
            for j in range(0,N):
                ddphi[i,j] = ((0.01111111*phi[mod(i+3,N),j]-0.15*phi[mod(i+2,N),j]+1.5*phi[mod(i+1,N),j]\
                               -2.72222222*phi[i,j]+1.5*phi[i-1,j]-0.15*phi[i-2,j]+0.01111111*phi[i-3,j])\
                               + (0.01111111*phi[i,mod(j+3,N)]-0.15*phi[i,mod(j+2,N)]+1.5*phi[i,mod(j+1,N)] \
                               -2.72222222*phi[i,j]+1.5*phi[i,j-1]-0.15*phi[i,j-2]+0.01111111*phi[i,j-3]))/(dx**2.0)
    return ddphi

# SPATIAL DERIVATIVE 3D (USE stencil=5)
@jit(nopython=True)
def Laplacian_3D(phi,dx,N):
    ddphi = zeros(shape=(N,N,N))
    for i in range (0,N):
        for j in range (0,N):
            for k in range (0,N):
                ddphi[i,j,k] = ((-phi[mod(i+2,N),j,k]+16*phi[mod(i+1,N),j,k]-30*phi[i,j,k]+16*phi[i-1,j,k] \
                                     - phi[i-2,j,k])+ (-phi[i,mod(j+2,N),k] + 16*phi[i,mod(j+1,N),k] -30*phi[i,j,k]\
                                    + 16*phi[i,j-1,k] - phi[i,j-2,k])+ (-phi[i,j,mod(k+2,N)] + 16*phi[i,j,mod(k+1,N)]\
                                    -30*phi[i,j,k] + 16*phi[i,j,k-1] - phi[i,j,k-2]))/(12*dx**2.0)
    return ddphi



# GRADIENT FIELD (2D for now)
@jit(nopython=True)
def Gradient_2D(f,dx,N):
    df = zeros(shape=(N,N))
    for i in range (0,N):
        for j in range (0,N):
            df[i,j] = ((- f[mod(i+2,N),j] + 8 * f[mod(i+1,N)] - 8 * f[i-1,j] + f[i-2,j])\
                      (- f[i,mod(j+2,N)] + 8 * f[i,mod(j+1,N)] - 8 * f[i,j-1] + f[i,j-2] ))/(12*dx)
    return df
