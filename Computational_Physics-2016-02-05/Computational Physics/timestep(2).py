import numpy as np
import force

def ts(r,v,acc,dt,ld,N):
    v += 0.5* acc * dt #halfway step for velocity (verlet)
    r += v * dt #update position
    #r=np.rint(r / ld) * ld
    r = np.mod(r , (ld)) #periodic boundary
    acc,V = force.FljArgon(r,ld,N) #forces due to new position
    v += 0.5 * acc * dt #complete velocity step
    
    return r,v,acc,V