import numpy as np
import scipy.sparse as sp
np.random.seed(12)
import warnings
import reg_class
from reg_class import * 
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import classymlp
#Comment this to turn on warnings
#warnings.filterwarnings('ignore')
### define Ising model aprams
# system size
L = 40
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

stateprod =  np.zeros((10000, L*L))
for i in range(10000):
    stateprod[i,:] = np.outer(states[i,:],states[i,:]).ravel()


energies=ising_energies(states,L)

##### calculate Ising energies 1-d

# comment out test_train_spilt method when doing neural network part. Affects data and output of model
"""
stateprod_train, stateprod_test, energies_train, energies_test= train_test_split(stateprod, energies, test_size = 0.5) 
test = regression(stateprod_train, energies_train)
Jtest = test.lasso(0.01)
pred = np.dot(stateprod_test, Jtest)
OLS_R2 = reg_class.R2(energies_test, pred) 
Jtest = Jtest.reshape((L,L))
import matplotlib.pyplot as plt
from matplotlib import cm
plt.imshow(Jtest,cmap = "seismic")
plt.show()
"""
##### end ising energies 1-d

###### shuffle data for neural network
order = list(range(np.shape(states)[0]))
np.random.shuffle(order)
states = states[order,:]
energies = energies[order]
valid_states = states[:100]
valid_energies = energies[:100]
states = np.delete(states, np.s_[:100], axis=0)
energies = np.delete(energies, np.s_[:100], axis=0)

train_states, test_states, energy_targets, energy_test = train_test_split(states, energies, test_size = 0.2) 
energy_targets = energy_targets[:, np.newaxis]
energy_test = energy_test[:, np.newaxis]
val_energies = valid_energies[:, np.newaxis]

eta = 0.001
hidden = 30
# make multi layer perceptron
network = classymlp.mlp(train_states, energy_targets, hidden, eta)
MSE, epochs = network.earlystopping(train_states, energy_targets, valid_states, val_energies)
MSE = np.array(MSE)
e_pred =  network.error(test_states, energy_test)
print('MSE on test data set is:', e_pred)

# must coincide with number of epochs in classymlp -> earlystopping 
# plot error
tot_epoch = np.linspace(1, epochs - 1, epochs - 1)
plt.semilogy(tot_epoch, MSE, label= 'Validation set')
plt.xlabel(r'Number of epochs', fontsize = 18)
plt.ylabel(r'MSE', fontsize = 18)
plt.legend()
plt.show()
