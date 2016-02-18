import numpy as np

def CV(lt,K,N):
    Knew = K[lt:]
    Kmean = np.mean(Knew)
    Kmean2 = Kmean*Kmean
    Kvar = np.var(Knew)
    Cv = (3*Kmean2)/(2*Kmean2 - 3*N*Kvar)
    return Cv