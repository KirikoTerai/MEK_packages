from MEK_finite_EB import * 
import math

import numpy as np
from numpy.linalg import eig

#packages for plotting
import matplotlib.pyplot as plt

N = 100
ztime = 10**(-12)
dt = 9/(N-1)

reservoir_rate = 10**5

DR_electron = 12
LR_electron = 0
HR_electron = 0
num_electron = DR_electron + LR_electron + HR_electron

initial_state = [DR_electron, 0, 0, 0, LR_electron, 0, 0, HR_electron]

## Inverted/Inverted
def EB_inv_inv_concerted(H1_redox, L1_redox, n):
    net = Network()
    # print(net.ET(-0.2, 10, net.reorgE, net.beta, net.V))
    slope = 0.15

    DR = Cofactor("DR", [-0.1]*num_electron)     # Two-electron midpoint potential
    D = Cofactor("D", [-0.4, 0.4])   #inverted
    L1 = Cofactor("L1", [L1_redox])
    L2 = Cofactor("L2", [-0.4 + slope*2])
    LR = Cofactor("LR", [-0.4 + slope*2]*num_electron)
    H1 = Cofactor("H1", [H1_redox])
    H2 = Cofactor("H2", [0.4 - slope*2])
    HR = Cofactor("HR", [0.4 - slope*2]*num_electron)

    net.addCofactor(DR, 0)
    net.addCofactor(D, 0)
    net.addCofactor(L1, 0)
    net.addCofactor(L2, 0)
    net.addCofactor(LR, 0)
    net.addCofactor(H1, 0)
    net.addCofactor(H2, 0)
    net.addCofactor(HR, 0)

    net.addConnection(DR, D, 10)
    net.addConnection(D, L1, 10)
    net.addConnection(D, L2, 20)
    net.addConnection(D, H1, 10)
    net.addConnection(D, H2, 20)
    net.addConnection(L1, L2, 10)
    net.addConnection(L2, LR, 10)
    net.addConnection(L1, H1, 20) 
    net.addConnection(L1, H2, 30) 
    net.addConnection(L2, H1, 30)
    net.addConnection(L2, H2, 40)
    net.addConnection(H1, H2, 10)
    net.addConnection(H2, HR, 10)

    net.set_Max_Electrons(num_electron)
    net.set_Min_Electrons(num_electron)

    net.constructStateList()

    net.constructAdjacencyMatrix()
    net.constructRateMatrix()

    # Make DR <-> D two-electron concerted step
    # Remove sequential electron transfer between DR <-> D
    net.two_electron_reservoir_step(DR, D, reservoir_rate)

    pop_MEK_init = np.zeros(net.adj_num_state)
    for allow_idx in net.allow:
        if net.state2idx(initial_state) == allow_idx:
            pop_MEK_init_idx = net.allow.index(allow_idx)
    pop_MEK_init[pop_MEK_init_idx] = 1

    if n == 0:
        t = 0
    else:
        t = ztime*(10**(n*dt))

    pop_MEK = net.evolve(t, pop_MEK_init)

    def get_2eFlux(cof: Cofactor, res:Cofactor, pop: np.array):
        flux = 0
        # Forward: cofactor -> reservoir
        site = cof
        res = res
        site_id = net.cofactor2id[site]
        res_id = net.cofactor2id[res]
        for initial in range(net.adj_num_state):
            for final in range(net.adj_num_state):
                if net.idx2state(net.allow[initial])[site_id]-net.idx2state(net.allow[final])[site_id]==2:
                    if net.idx2state(net.allow[final])[res_id]-net.idx2state(net.allow[initial])[res_id]==2:
                        # initial, final state found! check other electron conservation
                        I = np.delete(net.idx2state(net.allow[initial]), [site_id, res_id])     # Do not allow changes other than cof_i and cof_f we have searched for
                        J = np.delete(net.idx2state(net.allow[final]), [site_id, res_id])     # Deleting the cof_i and cof_f that is already checked to be correct
                        if np.array_equal(I, J):
                            kf = net.K[final][initial]
                            kb = net.K[initial][final]
                            # print(net.idx2state[initial], net.idx2state[final], kf, kb)
                            flux += pop[initial] * kf * 2
                            flux -= pop[final] * kb * 2
        return flux
    
    def get_1eFlux(cof: Cofactor, res:Cofactor, pop: np.array):
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
    
    flux_DR =  get_2eFlux(D, DR, pop_MEK)
    flux_HR =  get_1eFlux(H2, HR, pop_MEK)
    flux_LR =  get_1eFlux(L2, LR, pop_MEK)

    ## Productive flux
    firstET_flux = net.getCofactorFlux(D, 2, H1, 1, pop_MEK)
    secondET_flux = net.getCofactorFlux(D, 1, L1, 1, pop_MEK)
    ## Short-circuits
    SC1_flux = net.getCofactorFlux(D, 1, H1, 1, pop_MEK)
    SC2_flux = net.getCofactorFlux(L1, 1, D, 2, pop_MEK)

    popD = net.getExptvalue(pop_MEK, D)
    popH1 = net.getExptvalue(pop_MEK, H1)
    popH2 = net.getExptvalue(pop_MEK, H2)
    popL1 = net.getExptvalue(pop_MEK, L1)
    popL2 = net.getExptvalue(pop_MEK, L2)
    popDR = net.getExptvalue(pop_MEK, DR)
    popHR = net.getExptvalue(pop_MEK, HR)
    popLR = net.getExptvalue(pop_MEK, LR)

    # Total energy of the system
    net.makeEnergylist()
    # for i in range(net.adj_num_state):
    #     print(net.idx2state(net.allow[i]), net.Elist[i])
    energy = net.getEnergyvalue(pop_MEK)

    def deltaE(state_i, state_j):
        for i in range(net.adj_num_state):
            for j in range(net.adj_num_state):
                if net.idx2state(net.allow[i]) == state_i:
                    if net.idx2state(net.allow[j]) == state_j:
                        deltaE = net.Elist[j] - net.Elist[i]
        return deltaE

    deltaE_SC1 = deltaE([0, 1, 0, 0, 0, 0, 0, num_electron-1], [0, 0, 0, 0, 0, 1, 0, num_electron-1])
    deltaE_SC2 = deltaE([0, 1, 1, 0, 0, 0, 0, num_electron-2], [0, 2, 0, 0, 0, 0, 0, num_electron-2])
    deltaE_1st = deltaE([0, 2, 0, 0, 0, 0, 0, num_electron-2], [0, 1, 0, 0, 0, 1, 0, num_electron-2])
    deltaE_2nd = deltaE([0, 1, 0, 0, 0, 0, 0, num_electron-1], [0, 0, 1, 0, 0, 0, 0, num_electron-1])

    return flux_DR, flux_HR, flux_LR, t, popD, popH1, popH2, popL1, popL2, popDR, popHR, popLR, firstET_flux, secondET_flux, SC1_flux, SC2_flux, energy, deltaE_SC1, deltaE_SC2, deltaE_1st, deltaE_2nd

## Inverted/Normal
def EB_inv_norm_concerted(H1_redox, L1_redox, n):
    net = Network()
    slope = 0.15

    DR = Cofactor("DR", [-0.1]*num_electron)     # Two-electron midpoint potential
    D = Cofactor("D", [0.4, -0.4])   #normal
    L1 = Cofactor("L1", [L1_redox])
    L2 = Cofactor("L2", [-0.4 + slope*2])
    LR = Cofactor("LR", [-0.4 + slope*2]*num_electron)
    H1 = Cofactor("H1", [H1_redox])
    H2 = Cofactor("H2", [0.4 - slope*2])
    HR = Cofactor("HR", [0.4 - slope*2]*num_electron)

    net.addCofactor(DR, 0)
    net.addCofactor(D, 0)
    net.addCofactor(L1, 0)
    net.addCofactor(L2, 0)
    net.addCofactor(LR, 0)
    net.addCofactor(H1, 0)
    net.addCofactor(H2, 0)
    net.addCofactor(HR, 0)

    net.addConnection(DR, D, 10)
    net.addConnection(D, L1, 10)
    net.addConnection(D, L2, 20)
    net.addConnection(D, H1, 10)
    net.addConnection(D, H2, 20)
    net.addConnection(L1, L2, 10)
    net.addConnection(L2, LR, 10)
    net.addConnection(L1, H1, 20) 
    net.addConnection(L1, H2, 30) 
    net.addConnection(L2, H1, 30)
    net.addConnection(L2, H2, 40)
    net.addConnection(H1, H2, 10)
    net.addConnection(H2, HR, 10)

    net.set_Max_Electrons(num_electron)
    net.set_Min_Electrons(num_electron)

    net.constructStateList()

    net.constructAdjacencyMatrix()
    net.constructRateMatrix()

    net.two_electron_reservoir_step(DR, D, reservoir_rate)

    pop_MEK_init = np.zeros(net.adj_num_state)
    for allow_idx in net.allow:
        if net.state2idx(initial_state) == allow_idx:
            pop_MEK_init_idx = net.allow.index(allow_idx)
    pop_MEK_init[pop_MEK_init_idx] = 1

    if n == 0:
        t = 0
    else:
        t = ztime*(10**(n*dt))

    pop_MEK = net.evolve(t, pop_MEK_init)

    def get_2eFlux(cof: Cofactor, res:Cofactor, pop: np.array):
        flux = 0
        # Forward: D -> DR
        site = cof
        res = res
        site_id = net.cofactor2id[site]
        res_id = net.cofactor2id[res]
        for initial in range(net.adj_num_state):
            for final in range(net.adj_num_state):
                if net.idx2state(net.allow[initial])[site_id]-net.idx2state(net.allow[final])[site_id]==2:
                    if net.idx2state(net.allow[final])[res_id]-net.idx2state(net.allow[initial])[res_id]==2:
                        # initial, final state found! check other electron conservation
                        I = np.delete(net.idx2state(net.allow[initial]), [site_id, res_id])     # Do not allow changes other than cof_i and cof_f we have searched for
                        J = np.delete(net.idx2state(net.allow[final]), [site_id, res_id])     # Deleting the cof_i and cof_f that is already checked to be correct
                        if np.array_equal(I, J):
                            kf = net.K[final][initial]
                            kb = net.K[initial][final]
                            # print(net.idx2state[initial], net.idx2state[final], kf, kb)
                            flux += pop[initial] * kf * 2
                            flux -= pop[final] * kb * 2
        return flux
    
    def get_1eFlux(cof: Cofactor, res:Cofactor, pop: np.array):
        flux = 0
        # Forward: H2 -> HR
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
        
    flux_DR =  get_2eFlux(D, DR, pop_MEK)
    flux_HR =  get_1eFlux(H2, HR, pop_MEK)
    flux_LR =  get_1eFlux(L2, LR, pop_MEK)

    ## Productive flux
    firstET_flux = net.getCofactorFlux(D, 2, L1, 1, pop_MEK)
    secondET_flux = net.getCofactorFlux(D, 1, H1, 1, pop_MEK)
    ## Short-circuits
    SC1_flux = net.getCofactorFlux(D, 2, H1, 1, pop_MEK)
    SC2_flux = net.getCofactorFlux(L1, 1, D, 1, pop_MEK)

    popD = net.getExptvalue(pop_MEK, D)
    popH1 = net.getExptvalue(pop_MEK, H1)
    popH2 = net.getExptvalue(pop_MEK, H2)
    popL1 = net.getExptvalue(pop_MEK, L1)
    popL2 = net.getExptvalue(pop_MEK, L2)
    popDR = net.getExptvalue(pop_MEK, DR)
    popHR = net.getExptvalue(pop_MEK, HR)
    popLR = net.getExptvalue(pop_MEK, LR)

    # Total energy of the system
    net.makeEnergylist()
    # for i in range(net.adj_num_state):
    #     print(net.idx2state(net.allow[i]), net.Elist[i])
    energy = net.getEnergyvalue(pop_MEK)

    def deltaE(state_i, state_j):
        for i in range(net.adj_num_state):
            for j in range(net.adj_num_state):
                if net.idx2state(net.allow[i]) == state_i:
                    if net.idx2state(net.allow[j]) == state_j:
                        deltaE = net.Elist[j] - net.Elist[i]
        return deltaE

    deltaE_SC1 = deltaE([0, 2, 0, 0, 0, 0, 0, num_electron-2], [0, 1, 0, 0, 0, 1, 0, num_electron-2])
    deltaE_SC2 = deltaE([0, 0, 1, 0, 0, 0, 0, num_electron-1], [0, 1, 0, 0, 0, 0, 0, num_electron-1])
    deltaE_1st = deltaE([0, 2, 0, 0, 0, 0, 0, num_electron-2], [0, 1, 1, 0, 0, 0, 0, num_electron-2])
    deltaE_2nd = deltaE([0, 1, 0, 0, 0, 0, 0, num_electron-1], [0, 0, 0, 0, 0, 1, 0, num_electron-1])

    return flux_DR, flux_HR, flux_LR, t, popD, popH1, popH2, popL1, popL2, popDR, popHR, popLR, firstET_flux, secondET_flux, SC1_flux, SC2_flux, energy, deltaE_SC1, deltaE_SC2, deltaE_1st, deltaE_2nd

H1_redox = 0.25
L1_redox = -0.25

time = []
d_time_list = []

print("----------- Inverted/inverted (concerted) --------------")
flux_DR_ii_time = []
flux_HR_ii_time = []
flux_LR_ii_time = []
firstET_ii_time = []
secondET_ii_time = []
SC1_ii_time = []
SC2_ii_time = []
energy_ii_time = []
d_Etot_ii_time = []
deltaU_SC1_ii_time = []
deltaU_SC2_ii_time = []
deltaU_1st_ii_time = []
deltaU_2nd_ii_time = []
total_energy_ii_time = []
pop_D_ii_time = []
pop_H1_ii_time = []
pop_H2_ii_time = []
pop_L1_ii_time = []
pop_L2_ii_time = []
pop_DR_ii_time = []
pop_HR_ii_time = []
pop_LR_ii_time = []
for n in range(N):
    flux_DR_ii, flux_HR_ii, flux_LR_ii, t, popD_ii, popH1_ii, popH2_ii, popL1_ii, popL2_ii, popDR_ii, popHR_ii, popLR_ii, firstET_flux_ii, secondET_flux_ii, SC1_flux_ii, SC2_flux_ii, energy_ii, deltaE_SC1_ii, deltaE_SC2_ii, deltaE_1st_ii, deltaE_2nd_ii = EB_inv_inv_concerted(H1_redox, L1_redox, n)
    flux_DR_ii_time.append(flux_DR_ii)
    flux_HR_ii_time.append(flux_HR_ii)
    flux_LR_ii_time.append(flux_LR_ii)
    pop_D_ii_time.append(popD_ii)
    pop_H1_ii_time.append(popH1_ii)
    pop_H2_ii_time.append(popH2_ii)
    pop_L1_ii_time.append(popL1_ii)
    pop_L2_ii_time.append(popL2_ii)
    pop_DR_ii_time.append(popDR_ii)
    pop_HR_ii_time.append(popHR_ii)
    pop_LR_ii_time.append(popLR_ii)
    firstET_ii_time.append(firstET_flux_ii)
    secondET_ii_time.append(secondET_flux_ii)
    SC1_ii_time.append(SC1_flux_ii)
    SC2_ii_time.append(SC2_flux_ii)
    energy_ii_time.append(energy_ii)

    time.append(t)

    if n >= 10:
        d_time = time[n] - time[n-1]
        d_time_list.append(time[n])
        # Calculate first-order time derivative of E_tot
        E_tot_prev_ii = energy_ii_time[n-1]
        E_tot_curr_ii = energy_ii_time[n]
        d_Etot_dt_ii = (E_tot_curr_ii - E_tot_prev_ii)/d_time
        d_Etot_ii_time.append(d_Etot_dt_ii)
        # SC1: Energy dissipation per unit time 
        J_curr_SC1_ii = SC1_ii_time[n]
        deltaU_SC1_ii_time.append(deltaE_SC1_ii * J_curr_SC1_ii)
        # SC2: Energy dissipation per unit time 
        J_curr_SC2_ii = SC2_ii_time[n]
        deltaU_SC2_ii_time.append(deltaE_SC2_ii * J_curr_SC2_ii)
        # 1st ET: Energy dissipation per unit time 
        J_curr_1st_ii = firstET_ii_time[n]
        deltaU_1st_ii_time.append(deltaE_1st_ii * J_curr_1st_ii)
        # 2nd ET: Energy dissipation per unit time 
        J_curr_2nd_ii = secondET_ii_time[n]
        deltaU_2nd_ii_time.append(deltaE_2nd_ii * J_curr_2nd_ii)

        print("t=", t, E_tot_prev_ii, E_tot_curr_ii)

figii = plt.figure()
plt.plot(time, flux_DR_ii_time, color="darkviolet", label="DR")
plt.plot(time, flux_HR_ii_time, color="blue", label="HR")
plt.plot(time, flux_LR_ii_time, color="red", label="LR")
plt.xlabel("Time (s)", size="x-large")
plt.ylabel("Flux (s$^{-1}$)", size="x-large")
plt.xscale("log")
plt.yscale("symlog", linthresh=1e-2)
plt.legend()
figii.savefig("flux_ii_12e.pdf")

figii = plt.figure()
plt.plot(time, firstET_ii_time, color="orange", label="1st ET step")
plt.plot(time, secondET_ii_time, color="green", label="2nd ET step")
plt.plot(time, SC1_ii_time, color="black", label="SC1", linestyle="dashed")
plt.plot(time, SC2_ii_time, color="black", label="SC2", linestyle="dotted")
plt.xlabel("Time (s)", size="x-large")
plt.ylabel("Flux (s$^{-1}$)", size="x-large")
plt.xscale("log")
plt.yscale("symlog", linthresh=1e2)
# plt.ylim((-10**(1), 10**6))
plt.legend()
plt.tight_layout()
figii.savefig("productive_SC_flux_ii_12e.pdf")

figiip = plt.figure()
plt.plot(time, pop_D_ii_time, color="darkviolet", label="D")
plt.plot(time, pop_H1_ii_time, color="cyan", label="H1")
plt.plot(time, pop_H2_ii_time, color="blue", label="H2")
plt.plot(time, pop_L1_ii_time, color="salmon", label="L1")
plt.plot(time, pop_L2_ii_time, color="red", label="L2")
plt.plot(time, pop_DR_ii_time, color="darkviolet", label="DR", linestyle="dashed")
plt.plot(time, pop_HR_ii_time, color="blue", label="HR", linestyle="dashed")
plt.plot(time, pop_LR_ii_time, color="red", label="LR", linestyle="dashed")
plt.xlabel("Time (s)", size="x-large")
plt.ylabel("Expected number of electrons \n at cofactors", size="x-large")
plt.xscale("log")
plt.legend()
plt.tight_layout()
figiip.savefig("pop_ii_12e.pdf")

fig_e_SC = plt.figure()
plt.plot(d_time_list, deltaU_1st_ii_time, color="orange", label="1st ET step")
plt.plot(d_time_list, deltaU_2nd_ii_time, color="green", label="2nd ET step")
plt.plot(d_time_list, deltaU_SC1_ii_time, color="black", label="SC1", linestyle="dashed")
plt.plot(d_time_list, deltaU_SC2_ii_time, color="black", label="SC2", linestyle="dotted")
plt.plot(d_time_list, d_Etot_ii_time, color="red", label="dE$_{tot}$/dt")
plt.xlabel("Time (s)", size="x-large")
plt.ylabel("Energy dissipation \n per unit time (eV s$^{-1}$)", size="x-large")
plt.xscale("log")
plt.yscale("symlog", linthresh=1e2)
# plt.ylim((-10**(1), 10**6))
plt.legend()
plt.tight_layout()
fig_e_SC.savefig("energy_rate_ii_12e.pdf")

print("----------- Inverted/normal (concerted) --------------")
flux_DR_in_time = []
flux_HR_in_time = []
flux_LR_in_time = []
firstET_in_time = []
secondET_in_time = []
SC1_in_time = []
SC2_in_time = []
energy_in_time = []
d_Etot_in_time = []
deltaU_SC1_in_time = []
deltaU_SC2_in_time = []
deltaU_1st_in_time = []
deltaU_2nd_in_time = []
total_energy_in_time = []
pop_D_in_time = []
pop_H1_in_time = []
pop_H2_in_time = []
pop_L1_in_time = []
pop_L2_in_time = []
pop_DR_in_time = []
pop_HR_in_time = []
pop_LR_in_time = []
for n in range(N):
    flux_DR_in, flux_HR_in, flux_LR_in, t, popD_in, popH1_in, popH2_in, popL1_in, popL2_in, popDR_in, popHR_in, popLR_in, firstET_flux_in, secondET_flux_in, SC1_flux_in, SC2_flux_in, energy_in, deltaE_SC1_in, deltaE_SC2_in, deltaE_1st_in, deltaE_2nd_in = EB_inv_norm_concerted(H1_redox, L1_redox, n)
    flux_DR_in_time.append(flux_DR_in)
    flux_HR_in_time.append(flux_HR_in)
    flux_LR_in_time.append(flux_LR_in)
    pop_D_in_time.append(popD_in)
    pop_H1_in_time.append(popH1_in)
    pop_H2_in_time.append(popH2_in)
    pop_L1_in_time.append(popL1_in)
    pop_L2_in_time.append(popL2_in)
    pop_DR_in_time.append(popDR_in)
    pop_HR_in_time.append(popHR_in)
    pop_LR_in_time.append(popLR_in)
    firstET_in_time.append(firstET_flux_in)
    secondET_in_time.append(secondET_flux_in)
    SC1_in_time.append(SC1_flux_in)
    SC2_in_time.append(SC2_flux_in)
    energy_in_time.append(energy_in)

    if n >= 10:
        d_time = time[n] - time[n-1]
        # d_time_list.append(time[n])
        # Calculate first-order time derivative of E_tot
        E_tot_prev_in = energy_in_time[n-1]
        E_tot_curr_in = energy_in_time[n]
        d_Etot_dt_in = (E_tot_curr_in - E_tot_prev_in)/d_time
        d_Etot_in_time.append(d_Etot_dt_in)
        # SC1: Energy dissipation per unit time 
        J_curr_SC1_in = SC1_in_time[n]
        deltaU_SC1_in_time.append(deltaE_SC1_in * J_curr_SC1_in)
        # SC2: Energy dissipation per unit time 
        J_curr_SC2_in = SC2_in_time[n]
        deltaU_SC2_in_time.append(deltaE_SC2_in * J_curr_SC2_in)
        # 1st ET: Energy dissipation per unit time 
        J_curr_1st_in = firstET_in_time[n]
        deltaU_1st_in_time.append(deltaE_1st_in * J_curr_1st_in)
        # 2nd ET: Energy dissipation per unit time 
        J_curr_2nd_in = secondET_in_time[n]
        deltaU_2nd_in_time.append(deltaE_2nd_in * J_curr_2nd_in)

        print("t=", t, E_tot_prev_in, E_tot_curr_in)

figin = plt.figure()
plt.plot(time, flux_DR_in_time, color="darkviolet", label="DR")
plt.plot(time, flux_HR_in_time, color="blue", label="HR")
plt.plot(time, flux_LR_in_time, color="red", label="LR")
plt.xlabel("Time (s)", size="x-large")
plt.ylabel("Flux (s$^{-1}$)", size="x-large")
plt.xscale("log")
plt.yscale("symlog", linthresh=1e-2)
plt.legend()
figin.savefig("flux_in_12e.pdf")

figii = plt.figure()
plt.plot(time, firstET_in_time, color="orange", label="1st ET step")
plt.plot(time, secondET_in_time, color="green", label="2nd ET step")
plt.plot(time, SC1_in_time, color="black", label="SC1", linestyle="dashed")
plt.plot(time, SC2_in_time, color="black", label="SC2", linestyle="dotted")
plt.xlabel("Time (s)", size="x-large")
plt.ylabel("Flux (s$^{-1}$)", size="x-large")
plt.xscale("log")
plt.yscale("symlog", linthresh=1e2)
# plt.ylim((-10**(2), 10**9))
plt.legend()
plt.tight_layout()
figii.savefig("productive_SC_flux_in_12e.pdf")

figinp = plt.figure()
plt.plot(time, pop_D_in_time, color="darkviolet", label="D")
plt.plot(time, pop_H1_in_time, color="cyan", label="H1")
plt.plot(time, pop_H2_in_time, color="blue", label="H2")
plt.plot(time, pop_L1_in_time, color="salmon", label="L1")
plt.plot(time, pop_L2_in_time, color="red", label="L2")
plt.plot(time, pop_DR_in_time, color="darkviolet", label="DR", linestyle="dashed")
plt.plot(time, pop_HR_in_time, color="blue", label="HR", linestyle="dashed")
plt.plot(time, pop_LR_in_time, color="red", label="LR", linestyle="dashed")
plt.xlabel("Time (s)", size="x-large")
plt.ylabel("Expected number of electrons \n at cofactors", size="x-large")
plt.xscale("log")
plt.legend()
plt.tight_layout()
figinp.savefig("pop_in_12e.pdf")

fig_e = plt.figure()
plt.plot(time, energy_ii_time, color="black", label="Inverted")
plt.plot(time, energy_in_time, color="black", linestyle="dashed", label="Normal")
plt.xlabel("Time (s)", size="x-large")
plt.ylabel("Energy (eV)", size="x-large")
plt.xscale("log")
# plt.yscale("symlog", linthresh=1e2)
# plt.ylim((-10**(1), 10**6))
plt.legend()
plt.tight_layout()
fig_e.savefig("energy_12e.pdf")

fig_e_SC = plt.figure()
plt.plot(d_time_list, deltaU_1st_in_time, color="orange", label="1st ET step")
plt.plot(d_time_list, deltaU_2nd_in_time, color="green", label="2nd ET step")
plt.plot(d_time_list, deltaU_SC1_in_time, color="black", label="SC1", linestyle="dashed")
plt.plot(d_time_list, deltaU_SC2_in_time, color="black", label="SC2", linestyle="dotted")
plt.plot(d_time_list, d_Etot_in_time, color="red", label="dE$_{tot}$/dt")
plt.xlabel("Time (s)", size="x-large")
plt.ylabel("Energy dissipation \n per unit time (eV s$^{-1}$)", size="x-large")
plt.xscale("log")
plt.yscale("symlog", linthresh=1e2)
# plt.ylim((-10**(1), 10**6))
plt.legend()
plt.tight_layout()
fig_e_SC.savefig("energy_rate_in_12e.pdf")
