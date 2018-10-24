import numpy as np
import scipy.sparse as sp
np.random.seed(12)
import warnings
import reg_class
from reg_class import * 
from sklearn.model_selection import train_test_split

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

stateprod =  np.zeros((10000,L*L))
for i in range(10000):
    stateprod[i,:] = np.outer(states[i,:],states[i,:]).ravel()

print("før")
print(energies.shape, stateprod.shape)
stateprod_train, stateprod_test, energies_train, energies_test= train_test_split(stateprod,energies,test_size = 0.5) 
print("etter")
print(energies_test.shape, stateprod_test.shape, energies_train.shape, stateprod_train.shape)


test = regression(stateprod_train,energies_train)
Jtest = test.OLS()

Jtest = Jtest.reshape((L,L))
import matplotlib.pyplot as plt
from matplotlib import cm
plt.imshow(Jtest,cmap = "seismic")
plt.show()
