import numpy as np
from numba import njit
import time


def init_state(L,ordered=False):
    if ordered == True:
        state0 = np.ones((L,L))
    else:
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
    w = np.exp(-np.arange(-8,9,4)/temp)
    
    for i in range(1,cycles):
        Etemp = E[i-1]
        Mtemp = M[i-1]
        for j in range(L*L):
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
            if np.random.random() <= w[int(delta/4 + 2)]: #np.exp(-delta/temp):
                #print("pip")
                states[m,n] *= -1
                Etemp += delta
                Mtemp += 2*states[m,n]
                accept[i] += 1.
                #E2[i] = E[i]**2
        
        E[i] = Etemp
        M[i] = Mtemp
        #var_E = (E[i]**2 - E2[i])
    return E/L**2, np.abs(M/L**2), states, accept

if __name__ == "__main__":
    L = 20
    rand_state0,E0,M0 = init_state(L)
    rand_state1,E1,M1 = init_state(L)
    
    ordered_state0,E0_order,M0_order = init_state(L,ordered = True)
    ordered_state1,E1_order,M1_order = init_state(L,ordered = True)
    temp = 1
    start = time.time()
    cycles = int(5e4)
    #energies = 
    #magnetisations
    
    #for i in cycles = 
    energy1, magnet1,states1, accepted1= mcmc(rand_state0,E0,M0,cycles,temp)
    energy2, magnet2,states2, accepted2= mcmc(rand_state1,E1,M1,cycles,2.4)
    
    energy3, magnet3,states3, accepted3= mcmc(ordered_state0,E0_order,M0_order,cycles,temp)
    energy4, magnet4,states4, accepted4= mcmc(ordered_state1,E0_order,M0_order,cycles,2.4)
    
    stop = time.time()
    print(stop-start, "cycles" ,cycles)
    #%%
    import matplotlib.pyplot as plt
    
    if True: 
        #Energy plot
        plt.scatter(np.arange(cycles)[1::500],energy1[1::500],alpha=0.1,s =15)
        plt.scatter(np.arange(cycles)[::1000],energy3[::1001],alpha=0.1,s =15)
        plt.scatter(np.arange(cycles)[::200],energy2[::200],alpha=0.2,s =15)
        plt.scatter(np.arange(cycles)[::200],energy4[::200],alpha=0.2,s =15)
        
        plt.plot(np.arange(cycles),np.cumsum(energy1)/np.arange(1,cycles+1))
        plt.plot(np.arange(cycles),np.cumsum(energy3)/np.arange(1,cycles+1))
        plt.plot(np.arange(cycles),np.cumsum(energy2)/np.arange(1,cycles+1))
        plt.plot(np.arange(cycles),np.cumsum(energy4)/np.arange(1,cycles+1))      
        
        plt.ylabel("Energy per state",FontSize = 12)
    
    if False: 
        #Magnetization plot
        plt.scatter(np.arange(cycles)[1::1000],magnet1[1::1000],alpha=1,s =15)
        plt.scatter(np.arange(cycles)[::2000],magnet3[::2000],alpha=1,s =15)
        plt.scatter(np.arange(cycles)[::200],magnet2[::200],alpha=0.6,s =15)
        plt.scatter(np.arange(cycles)[::200],magnet4[::200],alpha=0.6,s =15)
        
        plt.plot(np.arange(cycles),np.cumsum(magnet1)/np.arange(1,cycles+1))
        plt.plot(np.arange(cycles),np.cumsum(magnet3)/np.arange(1,cycles+1))
        plt.plot(np.arange(cycles),np.cumsum(magnet2)/np.arange(1,cycles+1))
        plt.plot(np.arange(cycles),np.cumsum(magnet4)/np.arange(1,cycles+1)) 

        plt.ylabel("Magnetization",FontSize = 12)

    
    if False:
        #Accepted states plot
        plt.plot(np.arange(cycles),np.cumsum(accepted1))
        plt.plot(np.arange(cycles),np.cumsum(accepted3))
        plt.plot(np.arange(cycles),np.cumsum(accepted2))
        plt.plot(np.arange(cycles),np.cumsum(accepted4))
        plt.ylabel("Number of accepted States",FontSize = 12)
        plt.loglog()

    

    if True: 
        # common for all
        plt.legend(["T=1, Unordered","T=1, Ordered","T=2.4, Unordered","T=2.4, Ordered"])
        plt.xlabel("MC-cycles",FontSize = 12)
        plt.grid()
        plt.show()
    
    if False:
        #plt.subplot(121)
        #weights = np.ones_like(energy1[2500:]) / (len(energy1[2500:]))
        #plt.hist(energy1[2500:],bins=20,range=(-2,-1.9),weights = weights)
        #plt.title("T=1")
        plt.xlabel("Energy per spin", fontsize = 12)
        plt.ylabel("Probability of state",fontsize = 12)
        plt.subplot(122)
        weights = np.ones_like(energy2[2500:]) / (len(energy2[2500:]))
        plt.hist(energy2[2500:],bins=40,range=(-2,-.5),color="C2",weights = weights)
        plt.title("T=2.4",fontsize = 12)
    
        plt.xlabel("Energy per spin",fontsize = 12)
        plt.show()
