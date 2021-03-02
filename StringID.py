#===============================StringID.py============================#
# Created by Giovanni Pierobon 2021

# Description: Functions to identify, record and plot strings from the
# newtork evolution of an axion field 

#======================================================================#
from numpy import *
from numba import njit
import pandas as pd
from matplotlib import cm
import matplotlib.pyplot as plt

#########################################################
# Pi/2 method in 2D, requires 2/3 correction for 3D case
#########################################################

@njit
def cores_pi2(f,N,thr):
    accept = pi/2.0 - pi/2.0 *thr/100
    s=[]
    count=0
    for i in range(0,N-1):
        for j in range(0,N-1):
            south=abs(f[i+1][j]) - abs(f[i][j])
            east=abs(f[i+1][j+1]) - abs(f[i+1][j]) 
            north=abs(f[i][j+1]) - abs(f[i+1][j+1]) 
            west=abs(f[i][j]) - abs(f[i][j+1])
            if (south>accept or east>accept or north>accept or west>accept): 
                s.append([i,j])
                s.append([i,j+1])
                s.append([i+1,j])
                s.append([i+1,j+1])
    return int(len(s)/4.0)

#########################################################
# Pi method in 2D
#########################################################

# TO DO SOON 



#-----------------------------------------------------------------------------
# FINDING CORES AND PLOTTING 2D
def draw(f,N,index,size_x=13,size_y=12):
    s=[]
    count=0
    for i in range(0,N-1):
        for j in range(0,N-1):
            south=abs(f[i+1][j]) - abs(f[i][j]) # South side of (i,j) grid point
            east=abs(f[i+1][j+1]) - abs(f[i+1][j]) # East side of (i,j) grid point
            north=abs(f[i][j+1]) - abs(f[i+1][j+1]) # North side of (i,j) grid point
            west=abs(f[i][j]) - abs(f[i][j+1]) # West side of (i,j) grid point
            if (south>pi/2.0 or east>pi/2.0 or north>pi/2.0 or west>pi/2.0): 
                s.append([i,j])
                s.append([i,j+1])
                s.append([i+1,j])
                s.append([i+1,j+1])
    x_coord=[row[1] for row in s]
    y_coord=[row[0] for row in s]
    fig = plt.figure(figsize=(size_x,size_y))
    ax = fig.add_subplot(111)
    white_field=zeros(shape=(N,N))
    ax.imshow(white_field,origin='lower',cmap=cm.binary,vmax=1)
    plt.scatter(x_coord,y_coord,c=y_coord,cmap=cm.viridis,lw=1)
    ax.set_title(r'$\hat{\tau} = %d$' % index)
    return fig,ax,int(len(s)/4)







