import numpy as np
import math

def thermostat(v,N,T):
    v2 = np.multiply( v , v )
    K=np.sum(v2)
    scale=math.sqrt((N-1)*T*3/K)
    v=scale*v
    return v