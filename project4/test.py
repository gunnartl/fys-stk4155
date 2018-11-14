import numpy as np
from numba import njit
import time

L = 20

def init_state(L):    
    state0 = np.random.choice((-1,1),L*L).reshape((L,L))
    E0 = 0
    for i in range(L):
        for j in range(L):
            E0 += state0[i,j]*state0[i,j-1]+state0[i,j]*state0[i-1,j]
    return state0,E0

@njit
def mcmc(states,E0,cycles):
    E = np.zeros(cycles)
    E[0] = E0
    L = states.shape[0]
    for i in range(cycles):
        #flip random spin
        x = np.random.randint(L)
        y = np.random.randint(L)
        states[x,y] *= -1
        
        #energy_delta
        left = states[x-1,y]
        right = states[(x+1)%L,y]
        up = states[x,y-1]
        down = states[x,(y+1)%L]
        

        delta = 2*states[x,y] * (left+right+up+down)
        
        #check
        if delta > 0: 
            E[i] = E[i-1]+delta
        
        
    return E

state0,E0 = init_state(L)

start = time.time()
a = mcmc(state0,E0,int(1e1))
stop = time.time()
print(stop-start)