import numpy as np
import scipy.sparse as sp
np.random.seed(12)
import warnings
import reg_class
from reg_class import * 
#Comment this to turn on warnings
#warnings.filterwarnings('ignore')
### define Ising model aprams
# system size
L=40
# create 10000 random Ising states
states=np.random.choice([-1, 1], size=(10000,L))

def ising_energies(states,L):
    """
    This function calculates the energies of the states in the nn Ising Hamiltonian
    """
    J=np.zeros((L,L),)
    for i in range(L):
        J[i,(i+1)%L]-=1.0
        # compute energies
    E   = np.einsum("...i,ij,...j->...",states,J,states)
    E2  = np.einsum("...i,ij,...j",states,J,states)
    return E


# calculate Ising energies
energies=ising_energies(states,L)

s = np.outer(states[0,:],states[0,:]).ravel()

test = regression(s,energies[0])

#e =np.linalg.inv(s.T.dot(s)).dot(s.T.dot([[E[0]]]))
e= (1/(s.T.dot(s)))*s*(energies[0])
print(e)
#J = test.OLS()