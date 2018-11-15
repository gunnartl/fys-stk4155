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
def mcmc(states,E0,M0,cycles,temp):
    E = np.zeros(cycles)
    M = np.zeros(cycles)
    accept = np.zeros(cycles)
    E[0] = E0
    M[0] = M0
    L = states.shape[0]
    for i in range(1,cycles):
        #flip random spin
        m = np.random.randint(L)
        n = np.random.randint(L)
        #print(states)
        #print('')
        
        #energy_delta
        left = states[m-1,n]
        right = states[(m+1)%L,n]
        up = states[m,n-1]
        down = states[m,(n+1)%L]
        

        delta = 2*states[m,n] * (left+right+up+down)
        
        #check
        if np.random.random() <= np.exp(-delta/temp):
            #print("pip")
            states[m,n] *= -1
            E[i] = E[i-1]+delta
            M[i] = M[i-1]+2*states[m,n]
            accept[i] = 1.
            #E2[i] = E[i]**2
        else:
            E[i] = E[i-1]
            M[i] = M[i-1]
        #var_E = (E[i]**2 - E2[i])
    return E, M, states, accept

if __name__ == "__main__":
    L = 40
    rand_state0,E0,M0 = init_state(L)
    rand_state1,E1,M1 = init_state(L)
    
    ordered_state0 = np.ones((L,L))
    ordered_state0 = np.ones((L,L))
    temp = 1
    start = time.time()
    cycles = int(2e6)
    #energies = 
    #magnetisations
    
    #for i in cycles = 
    energy1, magnet1,states1, accepted1= mcmc(rand_state0,E0,M0,cycles,temp)
    energy2, magnet2,states2, accepted2= mcmc(rand_state1,E1,M1,cycles,2.4)
    
    stop = time.time()
    print(stop-start)
    #%%
    import matplotlib.pyplot as plt
    plt.plot(np.arange(cycles)[::1000]/400,energy1[::1000])
    plt.plot(np.arange(cycles)[::1000]/400,energy2[::1000])
    plt.legend(["T=1","T=2.4"])
    plt.xlabel("MC-cycles")
    plt.ylabel("# of accepted states")
    plt.grid()
    plt.show
    #fig,ax1 = plt.subplots()
    #ax1.plot(np.arange(cycles),energy1)
    #ax2 = ax1.twinx()
    #ax2.plot(np.arange(cycles),np.cumsum(accepted1))
    #plt.imshow(states1)
    #plt.show()