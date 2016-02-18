import numpy as np
from numba import jit
import math
def Forcesg(r, ld, N, bins, binlen):
    
    """
    compute forces on all particles. Also returns variables used for calculating correlation function (n) and pressure (force)
    r - Nx3 matrix containing x y z positions of all particles
    ld - length of the computional domain (for a box of V= ld**3)
    N - amount of particles
    bins - amount of bins for the correlation function
    binlen-length of the bin:ld/bins
    """
    #initiate vectors and 
    force=np.zeros(r.shape)
    #dr = math.sqrt(dx**2 + dy**2 + dz**2)
    epsilon = 1
    sigma = 1
    n=np.zeros(shape=(bins,))
    acc=np.zeros(shape=(N,3))
    pressure=0
    
    for i in range(N):
        for j in range(i):
            dx=r[i,0]-r[j,0]
            dy=r[i,1]-r[j,1]
            dz=r[i,2]-r[j,2]
            acc, force, dr2 = fastforceg(dx,dy,dz,acc,ld,i,j)
            distances=math.sqrt(dr2)
            n[int(distances/binlen)]+=2 #bin the distances of each particle
            pressure+=dr2*(-1*force) #virial component to calculate pressure
            
            
    return acc,n,pressure

@jit
def fastforceg(dx,dy,dz,acc,ld,i,j):
    """
    the just in time version that calculates the force of 1 particle on 1 other
    Returns the acc vector, containing the force in x y and z on each particle (shape N x 3)
    returns force for pressure 
    returns dr for correlation function
    """
    dx -= np.rint(dx / ld) * ld
    dy -= np.rint(dy / ld) * ld
    dz -= np.rint(dz / ld) * ld
    dr2= dx * dx + dy * dy + dz * dz
    drreturn = dr2
    force=0
    if dr2 < 6.25:
        dr2 = 1 / dr2
        dr6 = dr2 * dr2 * dr2
        dr12 = dr6 * dr6
        dr14 = dr12 * dr2
        dr8 = dr6 * dr2
        
        force = 24 * (2 * dr14 - dr8)
        acc[i,0] += force * dx
        acc[i,1] += force * dy
        acc[i,2] += force * dz
        acc[j,0] -= force * dx
        acc[j,1] -= force * dy
        acc[j,2] -= force * dz
        
    return acc, force, drreturn