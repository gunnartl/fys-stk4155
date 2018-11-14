import numpy as np
from numba import njit
import time


def init_state(L):    
    state0 = np.random.choice((-1,1),L*L).reshape((L,L))
    E0 = 0
    for i in range(L):
        for j in range(L):
            E0 -= state0[i,j]*state0[i,j-1]+state0[i,j]*state0[i-1,j]
    M0 = sum(state0)
    return state0,E0

@njit
def mcmc(states,E0,cycles,temp):
    E = np.zeros(cycles)
    #M = np.zeros(cycles)
    E[0] = E0
    accept = np.zeros(cycles)
    #M[0] = M0
    L = states.shape[0]
    for i in range(1,cycles):
        #flip random spin
        x = np.random.randint(L)
        y = np.random.randint(L)
        #print(states)
        #print('')
        
        #energy_delta
        left = states[x-1,y]
        right = states[(x+1)%L,y]
        up = states[x,y-1]
        down = states[x,(y+1)%L]
        

        delta = 2*states[x,y] * (left+right+up+down)
        
        #check
        if np.random.random() <= np.exp(-delta/temp):
            #print("pip")
            states[x,y] *= -1
            E[i] = E[i-1]+delta
            accept[i] = 1.
            #E2[i] = E[i]**2
        else:
            E[i] = E[i-1]
        #print(states)
        #var_E = (E[i]**2 - E2[i])
    return E, states, accept

if __name__ == "__main__":
    L = 20
    state0,E0 = init_state(L)
    temp = 2.4
    start = time.time()
    cycles = int(2e4)
    energy, states, accepted= mcmc(state0,E0,cycles,temp)
    
    stop = time.time()
    print(stop-start)
    
    import matplotlib.pyplot as plt
    plt.plot(np.arange(cycles),energy)
    plt.plot(np.arange(cycles),np.cumsum(accepted))
    plt.show