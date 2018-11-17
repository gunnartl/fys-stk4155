import numpy as np
from numba import njit, prange
import time

@njit
def init_state(state0):    
    #state0 = np.random.choice(np.array((-1,1)),L*L).reshape((L,L))
    L = state0.shape[0]
    E0 = 0
    for i in range(L):
        for j in range(L):
            E0 -= state0[i,j]*state0[i,j-1]+state0[i,j]*state0[i-1,j]
    M0 = np.sum(state0)
    return E0,M0

@njit(parallel = True)
def mcmc(L,cycles,temps):
    energies = np.zeros(len(temps))
    magnets = np.zeros(len(temps))
    sucepts = np.zeros(len(temps))
    heats = np.zeros(len(temps))
    for temp in prange(len(temps)):
        states = np.random.choice(np.array((-1,1)),L*L).reshape((L,L))
        E,M  = init_state(states)
        Eav  = 0#E
        Mav  = 0#M
        E2av = 0#E**2
        M2av = 0#M**2
        Mabs = 0#np.abs(M)

        #E2 = E**2
        #M2 = M**2
        w = np.exp(-np.arange(-8,9,4)/temps[temp])
        for i in range(cycles):
            #flip random spin
            for j in range(L*L):
                m = np.random.randint(L)
                n = np.random.randint(L)
                #compute delta
                left = states[m-1,n]
                right = states[(m+1)%L,n]
                up = states[m,n-1]
                down = states[m,(n+1)%L]
                
        
                delta = 2*states[m,n] * (left+right+up+down)
                
                #check
                if np.random.random() <= w[int(delta/4 + 2)]:
                    states[m,n] *= -1
                    E      += delta
                    M      += 2*states[m,n]
            if i>cycles/10:
                Eav += E
                Mav += M
                M2av += M**2
                E2av += E**2
                Mabs += np.abs(M)
                
                
        Eav  /= cycles
        E2av /= cycles
        M2av /= cycles
        Mav  /= cycles
        Mabs /= cycles
        heatcap = (E2av-Eav**2)/(temps[temp]**2)
        sucept  = (M2av-Mav**2)/(temps[temp])
        
        energies[temp] = Eav
        magnets[temp]  = Mabs
        heats[temp]    = heatcap
        sucepts[temp]  = sucept
    return energies/L**2,heats/L**2,magnets/L**2,sucepts/L**2

#%%
if __name__ == "__main__":
    L = 100
    temps = np.linspace(2.2,2.4,12)
    start = time.time()
    cycles = int(1e6)

    energy, Cv, Magnet, sucept = mcmc(L,cycles,temps)
    stop = time.time()
    print(stop-start)