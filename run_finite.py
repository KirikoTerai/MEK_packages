from MEK_finite import * 
import math

import numpy as np
from numpy.linalg import eig

#packages for plotting
import matplotlib.pyplot as plt

num_electron = 6    # Total number of electrons
initial_state = [num_electron, 0, 0, 0, 0]  # Example initial state. Initialize with all electrons placed in R1 reservoir

time = np.linspace(0, 0.01, 100)
popA_time = []
popB_time = []
popC_time = []
popR1_time = []
popR2_time = []
flux_R1_time = []
flux_R2_time = []
energy_time = []

net = Network()

# Define your cofactor
R1 = Cofactor("R1", [-0.2]*num_electron)   # This is a finite reservoir
A = Cofactor("A", [-0.1])   # Cofactor(name of cofactor, List of reduction potentials in eV)
B = Cofactor("B", [0])  # [1st reduction potential, 2nd reduction potential, ...]
C = Cofactor("C", [-0.2])    
R2 = Cofactor("R2", [-0.1]*num_electron)   # This is a finite reservoir

# Add your defined cofactors
# 0: Site that allows reversible hops, #1: trap site
net.addCofactor(R1, 0)
net.addCofactor(A, 0)
net.addCofactor(B, 0)
net.addCofactor(C, 0)
net.addCofactor(R2, 0)

# Define the physical connections
# Reservoir connections
net.addConnection(R1, A, 20)
net.addConnection(C, R2, 20)
# Cofactor connections
net.addConnection(A, B, 10)
net.addConnection(B, C, 10)  
 
# Remove microstates whose total number of electrons is not equal to num_electron
net.set_Max_Electrons(num_electron)
net.set_Min_Electrons(num_electron)

# Function to make all the accessible microstates
net.constructStateList()

net.constructAdjacencyMatrix()
# The dimension of the K matrix = number of microstates = net.adj_num_state
# K[j][i]: rate constant from microstate i -> j
net.constructRateMatrix()

for i in range(net.adj_num_state):
    # net.idx2state: maps the idx to the microstate 
    # net.state2idx: maps the microstate (list) to idx
    print(net.idx2state(net.allow[i]), i)

# Solve the master eq
# Initial condition of the microstate population
# System initiates with initial_state (defined at the beginning)
pop_MEK_init = np.zeros(net.adj_num_state)
for allow_idx in net.allow:
    if net.state2idx(initial_state) == allow_idx:
        pop_MEK_init_idx = net.allow.index(allow_idx)
pop_MEK_init[pop_MEK_init_idx] = 1

for t in time:
    pop_MEK = net.evolve(t, pop_MEK_init)
    
    # Expected number of electrons at each cofactor and reservoir
    popA = net.getExptvalue(pop_MEK, A)
    popB = net.getExptvalue(pop_MEK, B)
    popC = net.getExptvalue(pop_MEK, C)
    popR1 = net.getExptvalue(pop_MEK, R1)
    popR2 = net.getExptvalue(pop_MEK, R2)
    popA_time.append(popA)
    popB_time.append(popB)
    popC_time.append(popC)
    popR1_time.append(popR1)
    popR2_time.append(popR2)

    # Get reservoir flux
    def get_ReservoirFlux(cof: Cofactor, res:Cofactor, pop: np.array):
        flux = 0
        # Forward: cofactor -> reservoir
        site = cof
        res = res
        site_id = net.cofactor2id[site]
        res_id = net.cofactor2id[res]
        for initial in range(net.adj_num_state):
            for final in range(net.adj_num_state):
                if net.idx2state(net.allow[initial])[site_id]-net.idx2state(net.allow[final])[site_id]==1:
                    if net.idx2state(net.allow[final])[res_id]-net.idx2state(net.allow[initial])[res_id]==1:
                        # initial, final state found! check other electron conservation
                        I = np.delete(net.idx2state(net.allow[initial]), [site_id, res_id])     # Do not allow changes other than cof_i and cof_f we have searched for
                        J = np.delete(net.idx2state(net.allow[final]), [site_id, res_id])     # Deleting the cof_i and cof_f that is already checked to be correct
                        if np.array_equal(I, J):
                            kf = net.K[final][initial]
                            kb = net.K[initial][final]
                            # print(net.idx2state[initial], net.idx2state[final], kf, kb)
                            flux += pop[initial] * kf
                            flux -= pop[final] * kb
        return flux
    
    flux_R1 = get_ReservoirFlux(A, R1, pop_MEK)
    flux_R2 = get_ReservoirFlux(C, R2, pop_MEK)
    flux_R1_time.append(flux_R1)
    flux_R2_time.append(flux_R2)

    # Get total energy of the system
    net.makeEnergylist()    # Makes an energy list: [Energy of microstate 1, energy of microstate 2, ....]
    # for i in range(net.adj_num_state):
    #     print(net.idx2state(net.allow[i]), net.Elist[i])
    energy = net.getEnergyvalue(pop_MEK)
    energy_time.append(energy)


fig = plt.figure()
plt.plot(time, popA_time, color="black", label="A")
plt.plot(time, popB_time, color="red", label="B")
plt.plot(time, popC_time, color="blue", label="C")
plt.plot(time, popR1_time, color="black", label="R1", linestyle="dashed")
plt.plot(time, popR2_time, color="blue", label="R2", linestyle="dashed")
plt.xlabel("Time (s)", size="x-large")
plt.ylabel("Population", size="x-large")
plt.legend()
fig.savefig("run_finite_pop.pdf")

fig = plt.figure()
plt.plot(time, flux_R1_time, color="black", label="R1")
plt.plot(time, flux_R2_time, color="blue", label="R2")
plt.xlabel("Time (s)", size="x-large")
plt.ylabel("Flux (s$^{-1}$)", size="x-large")
plt.legend()
plt.yscale("symlog")
plt.tight_layout()
fig.savefig("run_finite_flux.pdf")

fig = plt.figure()
plt.plot(time, energy_time, color="black")
plt.xlabel("Time (s)", size="x-large")
plt.ylabel("Total energy of the system (eV)", size="x-large")
plt.tight_layout()
fig.savefig("run_finite_energy.pdf")
