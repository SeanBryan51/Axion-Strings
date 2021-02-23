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

# TIME EVOLUTION (LEAPFROG VELOCITY VERLET ALGORITHM)
@jit(nopython=True)
def Evolve(phi1,phi2,phidot1,phidot2,f1,f2,Laplacian_2D,alpha,era,lambdaPRS,N,dx,dt,t):
    phi1 = phi1 + dt*(phidot1 + 0.5*f1*dt)
    phi2 = phi2 + dt*(phidot2 + 0.5*f2*dt)
    # UPDATE
    t = t+dt
    f1_next = Laplacian_2D(phi1,dx,N) - 2*alpha*(era/t)*phidot1 - lambdaPRS*phi1*(phi1**2.0+phi2**2.0 - 1)# + 10**5.0/(t**2.0) 
    f2_next = Laplacian_2D(phi2,dx,N) - 2*alpha*(era/t)*phidot2 - lambdaPRS*phi2*(phi1**2.0+phi2**2.0 - 1)# + 10**5.0/(t**2.0) 
    #KICK
    phidot1 = phidot1 + 0.5*(f1 + f1_next)*dt
    phidot2 = phidot2 + 0.5*(f2 + f2_next)*dt
    # UPDATE
    f1 = 1.0*f1_next
    f2 = 1.0*f2_next
    return phi,phi2