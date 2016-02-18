# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 14:39:07 2016

@author: rubenbiesheuvel

This program is for computing the interaction between argon particles in a Leonard Jhones potential , with periodic boundary conditions 
and initial conditions.

The steps we have to take are the following:

Initialize:
    import libraries
    Initial conditions for speed and positions
    boundary conditions
    
For each timestep:
    Calculate the force on each particle, dependent on the Leonard Jones potential
    Calculate the path the are going ot take using the Velocity-verlet
    Store the new place and velocity (velocity for temperature calculations)
"""

import numpy as np
import math as math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import maxwell

## Setting all the given entries of the problem
T = 3 # initial temperature in Kelvin
L = 3 #number of unit cells in 3 directions
Z = 4 #number of atoms per unit cell
N = Z*L**3 #number of atoms in total space 
r = np.zeros(shape=(N,3), dtype="float64")
v = np.zeros(shape=(N,3), dtype = "float64")
l = 1 #size of unit cell in this case

M = 39.948 * 1.660538921*10**(-27) #mass argon in kg
k = 1.38064852*10**(-23) #Boltzmann constant in SI units

## Computing the initial positions
p1 = np.array([0.25, 0.25, 0.25])
p2 = np.array([0.75, 0.75, 0.25])
p3 = np.array([0.75, 0.25, 0.75])
p4 = np.array([0.25, 0.75, 0.75])

n=0     
for h in range(L):
    for i in range(L):
        for j in range(L):
            disp = np.multiply(l,np.array([j,i,h])) #displacement array
            r[n] = p1 + disp
            r[n+1] = p2+ disp
            r[n+2] = p3 + disp
            r[n+3] = p4 + disp
            n+=4
            
 ## Plotting a scatter gragh of the initial positions on a fcc configuration           
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(r[:,0],r[:,1],r[:,2])

## Computing of the initial velocities trough a Maxwell Distribution
sigma = math.sqrt(k*T/M) #variance of the system
mu = 0  #mean speed
s = np.random.normal(mu, sigma, 500)

for i in range(3):
    v[:,i]= np.random.normal(mu, sigma, N) # creating a samples from a gaussian distribution 
    mean = np.mean(v[:,i])
    v[:,i] = v[:,i] - mean
    
 ## Plotting a histogram of the velocities   
fig = plt.figure()
plt.hist(v)
    
 ## Boundary conditions
    

