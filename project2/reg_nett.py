import numpy as np
import reg_class
from reg_class import * 
from sklearn.model_selection import train_test_split
import time

#np.random.seed(12)
L=40

states=np.random.choice([-1, 1], size=(10000,L))
#bias_states = np.c_[np.ones(10000),states]

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



#def relu(x):
#    if x>

# calculate Ising energies
energies=ising_energies(states,L)[:,np.newaxis]

states, test_states, energies, test_energies = train_test_split(states,energies,train_size=0.8)

def feed_forward(inputs,weights,biases):
    
    Layer = inputs @ weights[0] +biases[0]
    
    siglayer = 1/(1+np.exp(-Layer))
    
    
    E = np.dot(siglayer,weights[1])+biases[1]
    
    return E, siglayer


def back_propagation(target, output, weights,siglayer,biases,eta,inputs):
    e_out    = output-target
    e_hidden = e_out @ weights[1].T * siglayer * (1-siglayer)

    weights[1] -= eta * siglayer.T @ e_out
    weights[0] -= eta* inputs.T @ e_hidden
    
    biases[1] -= eta * np.sum(e_out,axis = 0)
    biases[0] -= eta * np.sum(e_hidden, axis = 0)
    return weights , biases
    

n_hidden  = 100
eta = 0.001
    
w1 = np.random.randn(states.shape[1],n_hidden)/100
w2 = np.random.randn(n_hidden,1)/100
weights = np.array([w1,w2])
b1 = np.zeros(n_hidden)+np.random.random()/10
b2 = np.random.random()/10
biases = np.array([b1,b2])

a = []

batch = 100

start = time.time()
for i in range(50000):
    index = np.random.randint(8000,size=batch)
    E,siglayer = feed_forward(states[index],weights,biases)
    weights,biases = back_propagation(energies[index],E,weights,siglayer,biases,eta,states[index])
    #print(reg_class.MSE(E,energies[index]))
    a.append(reg_class.MSE(energies[index],E))

stop = time.time()
print(stop-start)
import matplotlib.pyplot as plt
plt.plot(np.arange(len(a[12:]))/batch,a[12:])
plt.semilogy()
plt.xlabel("Epochs")
plt.ylabel("MSE")

test_E, repp = feed_forward(test_states,weights,biases)
print(reg_class.MSE(test_energies,test_E))
print(max(abs(test_energies)-abs(test_E)))
