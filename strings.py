from numpy import *
from numba import njit

@njit # This computes the for loops in machine code for faster running 
def cores(f,N): # f would be the axion field at the given time step
    s=[]
    count=0
    for i in range(0,N-1):
        for j in range(0,N-1):
            south=abs(f[i+1][j])-abs(f[i][j]) # South side of (i,j) grid point
            east=abs(f[i+1][j+1])-abs(f[i+1][j]) # East side of (i,j) grid point
            north=abs(f[i][j+1])-abs(f[i+1][j+1]) # North side of (i,j) grid point
            west=abs(f[i][j])-abs(f[i][j+1]) # West side of (i,j) grid point
            if (abs(south)>pi/2.0 or abs(east)>pi/2.0 or abs(north)>pi/2.0 or abs(west)>pi/2.0): 
                s.append([i,j])
    for a in range(0,len(s)-1):
        diff_y=s[a+1][1]-s[a][1]
        diff_x=s[a+1][0]-s[a][0]
        if (diff_y == 0 and diff_x == 1):
            count+=1
        if (diff_y == 1 and diff_x == 0):
            count+=1
    num=len(s)-count
    return num

@njit
def cores2(f,N): 
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
    return int(len(s)/4.0)