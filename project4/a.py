import numpy as np
from numba import njit
import time


def init_state(L):    
    state0 = np.random.choice((-1,1),L*L).reshape((L,L))
    E0 = 0
    for i in range(L):
        for j in range(L):
            E0 -= state0[i,j]*state0[i,j-1]+state0[i,j]*state0[i-1,j]
    M0 = np.sum(state0)
    return state0,E0,M0

@njit
def mcmc(states,E,M,cycles,temp):
    Eav  = E
    Mav  = M
    E2av = E**2
    M2av = M**2
    Mabs = np.abs(M)
    L = states.shape[0]
    for j in range(cycles):
        for i in range(L**2):
            #flip random spin
            m = np.random.randint(L)
            n = np.random.randint(L)
    
            #compute delta
            left = states[m-1,n]
            right = states[(m+1)%L,n]
            up = states[m,n-1]
            down = states[m,(n+1)%L]
            
    
            delta = 2*states[m,n] * (left+right+up+down)
            
            #check
            if np.random.random() <= np.exp(-delta/temp):
                states[m,n] *= -1
                E      += delta
                M      += 2*states[m,n]
                #E2     += E**2
                #M2     += M**2
        
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
    
    heatcap = (E2av-Eav**2)/(temp**2)
    sucept  = (M2av-Mav**2)/temp
    return Eav/L**2,heatcap/L**2,Mabs/L**2,sucept/L**2

if __name__ == "__main__":
    L = 2

    #state1,E1,M1 = init_state(L)
    temp = 1 # temperature
    start = time.time()
    #cycles=int(1e7)
    cycles = 10**np.array([2,3,4,5,6,7])
    energy = np.zeros(len(cycles))
    magnet = np.zeros(len(cycles))
    Cv     = np.zeros(len(cycles))
    sucept = np.zeros(len(cycles))
    
    #Checks energy, heatcapacity, megnetisation and suceptebility for several 
    #numbers of cycles
    for i,j in enumerate(cycles):
        state0,E0,M0 = init_state(L)
        energy[i], Cv[i], magnet[i],sucept[i]= mcmc(state0,E0,M0,j,temp)


    
    stop = time.time()
    
    np.savetxt("tabell_a.txt",np.c_[cycles,energy,magnet,Cv,sucept]) # saves a NEAT table
    print("time",stop-start)