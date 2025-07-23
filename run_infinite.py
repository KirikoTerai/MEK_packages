from MEK_infinite import * 
import math

import numpy as np
from numpy.linalg import eig

#packages for plotting
import matplotlib.pyplot as plt

reservoir_rate = 10**2
time = np.linspace(0, 0.01, 100)

popA_time = []
popB_time = []
popC_time = []
flux_R1_time = []
flux_R2_time = []

for t in time:
    net = Network()

    # Define your cofactor
    A = Cofactor("A", [-0.1])   # Cofactor(name of cofactor, List of reduction potentials in eV)
    B = Cofactor("B", [0])  # [1st reduction potential, 2nd reduction potential, ...]
    C = Cofactor("C", [-0.2])

    # Add your defined cofactors
    net.addCofactor(A)
    net.addCofactor(B)
    net.addCofactor(C)

    # Define the physical connections
    net.addConnection(A, B, 10)
    net.addConnection(B, C, 10)   

    # Add infinite reservoirs
    """
    net.addReservoir(name, cofactor, redox_state, num_electron, deltaG, reservoir_rate)
    name -- name of reservoir
    cofactor -- cofactor connected to reservoir
    redox_state -- redox state of the cofactor when it interacts with the reservoir
    num_electron -- number of electrons that move when the cofactor interacts with the reservoir
    deltaG -- deltaG for electron transfer from cofactor -> reservoir
    reservoir_rate -- electron transfer rate from cofactor -> reservoir
    """
    net.addReservoir("R1", A, 1, 1, 0.1, reservoir_rate)
    net.addReservoir("R2", C, 1, 1, -0.1, reservoir_rate)

    # Function to make all the accessible microstates
    net.constructStateList()

    net.constructAdjacencyMatrix()
    # The dimension of the K matrix = total number of microstates = net.adj_num_state
    # K[j][i]: rate constant from microstate i -> j
    net.constructRateMatrix()

    for i in range(net.adj_num_state):
        # net.idx2state: maps the idx to the microstate (list)
        # net.state2idx: maps the microstate (list) to idx
        print(net.idx2state(i), i)

    # Solve the master eq
    # Initial condition of the microstate population
    pop_MEK_init = np.zeros(net.num_state)
    pop_MEK_init[0] = 1     # System initiates with [0,0,0]

    pop_MEK = net.evolve(t, pop_MEK_init)
    
    # Population at each cofactor
    popA = net.getExptvalue(pop_MEK, A)
    popB = net.getExptvalue(pop_MEK, B)
    popC = net.getExptvalue(pop_MEK, C)
    popA_time.append(popA)
    popB_time.append(popB)
    popC_time.append(popC)

    # Reservoir fluxes
    flux_R1 = net.getReservoirFlux("R1", pop_MEK)
    flux_R2 = net.getReservoirFlux("R2", pop_MEK)
    flux_R1_time.append(flux_R1)
    flux_R2_time.append(flux_R2)

fig = plt.figure()
plt.plot(time, popA_time, color="black", label="A")
plt.plot(time, popB_time, color="red", label="B")
plt.plot(time, popC_time, color="blue", label="C")
plt.xlabel("Time (s)", size="x-large")
plt.ylabel("Population at cofactors", size="x-large")
plt.legend()
plt.tight_layout()
fig.savefig("run_infinite_pop.pdf")

fig = plt.figure()
plt.plot(time, flux_R1_time, color="black", label="R1")
plt.plot(time, flux_R2_time, color="blue", label="R2")
plt.xlabel("Time (s)", size="x-large")
plt.ylabel("Flux (s$^{-1}$)", size="x-large")
plt.legend()
plt.yscale("symlog")
plt.tight_layout()
fig.savefig("run_infinite_flux.pdf")
