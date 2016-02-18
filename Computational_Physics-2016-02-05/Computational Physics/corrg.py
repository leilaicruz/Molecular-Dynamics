import numpy as np
import math
from numba import jit

@jit
def correlation(VN,r,maxs,dmaxs,bins,N,ld):
    R = np.zeros(shape=(N,N))
    for i in range(N):
        for j in range(i):
            dx=r[i,0]-r[j,0]
            dy=r[i,1]-r[j,1]
            dz=r[i,2]-r[j,2]
            dx -= np.rint(dx / ld) * ld
            dy -= np.rint(dy / ld) * ld
            dz -= np.rint(dz / ld) * ld        
            drr = math.sqrt( dx * dx + dy * dy + dz * dz)
            R[i,j] = drr
            R[j,i] = drr

    #bins = 30
    #maxs = np.linspace(0,ld,num=bins)
    #dmaxs = maxs[1] - maxs[0]
    g = np.zeros(shape=(bins,))
    for i in range(1,bins):
        maxx = maxs[i]
        for j in range(N):
            for k in range(j):
                dist = 1/(4*math.pi*R[j,k]*R[j,k]*dmaxs)
                x = R[j,k] - maxx
                if (-1*dmaxs < (x) <0):
                    g[i-1] += 2*dist
            
    g = g*VN
    return g