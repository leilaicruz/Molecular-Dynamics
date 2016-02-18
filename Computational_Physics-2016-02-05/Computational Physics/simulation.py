import numpy as np
import timestep
from thermostat import thermostat
from corrg import correlation

def simulate(Time,dt,r,v,acc,V,ld,N,T,lt,bins,VN,maxs,dmaxs):
    #Do the whole simulation, to provide the total Kinetic energy for all the timesteps

    #initialize vectors
    Energy = np.zeros(shape=(Time, ))
    Ve = np.zeros(shape=(Time, ))
    K = np.zeros(shape=(Time, ))
    g = np.zeros(shape=(bins,))
    
    for i in range(Time):
        r, v, acc, V = timestep.ts(r,v,acc,dt,ld,N)
        #fraction of timesteps for changing velocity with thermostat
    
        if i<lt and (np.mod(i,lt/200)==0): #mod for calling thermostat 10 times in time lt
            v = thermostat(v,N,T)
        v2 = np.multiply(v,v)
        K[i] = 0.5*np.sum(v2) #kinetic energy
        Ve[i]= V #potential energy
        Energy[i] = Ve[i] + K[i] #total energy
        if i>lt:
            g += correlation(VN,r,maxs,dmaxs,bins,N,ld)
        
    return K, Energy, r, g