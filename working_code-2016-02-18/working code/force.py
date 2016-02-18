import numpy as np
from numba import jit
import math

def Forces(r, ld, N, bins,binlen):
    force=np.zeros(r.shape)
    n = np.zeros(shape=(bins,))
    #dr = math.sqrt(dx**2 + dy**2 + dz**2)
    epsilon = 1
    sigma = 1
    
    acc=np.zeros(shape=(N,3))
    V=0
    
    for i in range(N):
        for j in range(i):
            dx=r[i,0]-r[j,0]
            dy=r[i,1]-r[j,1]
            dz=r[i,2]-r[j,2]
            acc, V = fastforce(dx,dy,dz,V,acc,ld,i,j)
            
    return acc, V

@jit
def fastforce(dx,dy,dz,V,acc,ld,i,j):
    dx -= np.rint(dx / ld) * ld
    dy -= np.rint(dy / ld) * ld
    dz -= np.rint(dz / ld) * ld
    dr2= dx * dx + dy * dy + dz * dz
    if dr2 < 6.25:
        dr2 = 1 / dr2
        dr6 = dr2 * dr2 * dr2
        dr12 = dr6 * dr6
        dr14 = dr12 * dr2
        dr8 = dr6 * dr2
        V += 4 * (dr12 - dr6)
        force = 24 * (2 * dr14 - dr8)
        acc[i,0] += force * dx
        acc[i,1] += force * dy
        acc[i,2] += force * dz
        acc[j,0] -= force * dx
        acc[j,1] -= force * dy
        acc[j,2] -= force * dz
        
    return acc, V