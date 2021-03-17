'''
Monte Carlo Simulation for an Einstein Solid

'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from EinsteinSolid import *

def temp(N, Q):
    '''
    Calculates temperature using this equation:
    a{temp} = 1 / (1 + (N / Q))

    Parameters:
    N: number of oscillators
    Q: Total number of quanta

    returns temperature
    '''
    return 1 / np.log(1 + (Q / N))

def boltzWeights(Q, N):
    '''
    
    Parameters:
    n_max: array of inteegers for dimensionless quantity kT/epsilon0
    q: quanta per oscillator

    returns  initial n values and analytical solution
    '''

    #Creating an array for n_max (n_vals )
    n_vals = np.arange(Q + 1)
    
    #Creating array of these values using this: e^(-n/alpha)
    e_vals = np.exp((-1 * (n_vals)) / temp(Q, N))

    #Calculating normalizing constant:
    norm_const = np.sum(e_vals)
    
    #returning analytical solution
    return (n_vals, e_vals/norm_const)


#PLOTTING GRAPH:

#Setting up plot:
fig1, graph1 = plt.subplots(figsize = (10, 10))

# Generate a Monte Carlo Sample of the Einstein Solid

# Physical Parameters
L = 16      # Square root of the # of oscillators
N = L * L   # number of oscillators
q = np.array([0.5, 1, 2, 4])       # quanta per oscillator
Q = q * N   # Total number of quanta

# Simulation Parameters
EXCHANGES_PER_UPDATE = Q  # Number of exchanges per update
NUM_SAMPLES = 100         # Number of samples to generate

#Iterating thru q values
for i in range(len(q)):
    #Creating new Einstein Solid object
    einsol = EinsteinSolid(N, q[i], EXCHANGES_PER_UPDATE[i])
    
    #Perform Monte Carlo updates, then create a plot using this
    einsol.plotProbDist(einsol.sampleProbDist(NUM_SAMPLES), errorbar = True, custom_graph = True, set_graph = graph1)

    #Getting data for analytical solution:
    #n should vary from 0 to Q
    data1 = boltzWeights(Q[i], N)

    #Plotting
    graph1.semilogy(data1[0], data1[1], label = "q = " + str(q[i]))

#Setting up graph
graph1.set_ylabel("P(n)", fontsize = 16)
graph1.set_xlabel("n", fontsize = 16)
graph1.set_title("P(n) vs. n", fontsize = 20)
graph1.legend()
graph1.set_ylim(10**-6, 1)
graph1.set_xlim(-1, 50)
plt.show()