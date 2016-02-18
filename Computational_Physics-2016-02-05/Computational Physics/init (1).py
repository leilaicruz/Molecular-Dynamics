import numpy as np
import math
from itertools import product
# coding: utf-8

# In[ ]:

def position(l,N,L):
    # particles in unit cell
    r = np.zeros(shape=(N, 3), dtype="float64")
    #coordinates of 4 particles in the unit cell
    p1 = l*np.array([0.25, 0.25, 0.25])
    p2 = l*np.array([0.75, 0.75, 0.25])
    p3 = l*np.array([0.75, 0.25, 0.75])
    p4 = l*np.array([0.25, 0.75, 0.75])

    # distribute all the particles by using the unit cell and displacing it in x y and z with length l
    n=0
    for x, y, z in product(range(L), range(L), range(L)):
        disp = np.multiply(l, np.array([x, y, z])) #displacement array
        r[n] = p1 + disp
        r[n + 1] = p2 + disp
        r[n + 2] = p3 + disp
        r[n + 3] = p4 + disp
        n += 4
        
    return r

def velocity(T,N):
    v = np.zeros(shape=(N, 3), dtype="float64")
    sigma = math.sqrt(T) #variance of the system
    mu = 0 #mean speed
    v = np.random.normal(mu, sigma, 3*N).reshape(-1, 3)
    v -= v.sum(axis=0) / N
    return v

