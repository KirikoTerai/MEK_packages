import numpy as np
import math
import matplotlib.pyplot as plt
from scipy import linalg
from collections import defaultdict as Dict


class Cofactor():
    def __init__(self, name: str, redox: list):
        """
        Initialize this cofactor object, with property: name, and redox potentials
        Arguments:
            name {str} -- Name of the cofactor
            redox {list} -- List of ORDERED redox potential for different redox states
        """
        self.name = name
        self.redox = redox          #(ex.) "[first reduction potential (0 -> 1), second reduction potential (1 -> 2),...]
        self.capacity = len(redox)    # The number of electrons the site can occupy is equal to the number of reduction potentials

    def __str__(self) -> str:         #__str__:a built-in function that computes the "informal" string representations of an object
        """
        Return a string representation of the cofactor
        Returns:
            str -- String representation of the cofactor
        """
        s = ""
        # Initialize with cofactor name
        s += "Cofactor Name: {}\n".format(self.name)     #\n:new line in string
        s += "------------ \n"     #Draw a line between cofactor info (looks cuter!)
        # Print cofactor info, with state_id and relative redox potential
        for i in range(len(self.redox)):
            s += "Redox State ID: {}, Oxidation Potential: {}\n".format(i, self.redox[i])

        return s
    
class Network():
    def __init__(self):
        """
        Initialize the whole system
        NOTICE: the initialized Network instance has nothing in it, use other functions to insert information
        """
        # system-specific data structure and parameters
        self.num_cofactor = 0
        self.num_state = 1
        self.adj_num_state = 1 # adjusted number of states with max particle ceiling
        self.allow = [] # list of allowed states with max particle ceiling
        self.Elist = [] 
        self.potentiallist = []
        self.Midptlist = []
        self.id2cofactor = dict()  # key-value mapping is id-cofactor
        self.cofactor2id = dict()  # key-value mapping is cofactor-id
        self.adjacencyList = list()
        self.D = None  # not defined
        self.K = None  # not defined
        self.siteCapacity = []  # indexing is id-site_capacity
        self.num_reservoir = 0
        self.reservoirInfo = dict()    # key-value mapping is id-reservoir name, cofactor, redox_state, num_electron, deltaG, rate
        self.id2reservoir=dict()    # key-value mapping is id-reservoir name
        self.reservoir2id=dict()    # key-value mapping is reservoir name-id
        self.max_electrons = None
        self.min_electrons = None
        self.num_sink = 0
        self.sink_rate = None   # The electron transfer rate into a sink. The rate to go out from the sink is 0.
        self.id2sink = dict()  # key-value mapping is id-sink
        self.sink2id = dict()  # key-value mapping is sink-id
        """
        ET-specific data structure and parameters     #Incorporate to the ET function?
        """
        self.hbar = 6.5821 * 10 ** (-16)  # unit: eV sec
        self.beta = 39.06  # unit: 1/kT in 1/eV (room temperature)
        self.reorgE = 0.7 # unit: eV
        self.V = 0.1 # unit: eV

    def __str__(self) -> str:
        """
        Return a string representation of the defined Network
        Returns:
            str -- String representation of the Network
        """
        s = ""
        # 1. print Cofactors information
        s += "Total number of cofactors in the Network: {}\n".format(self.num_cofactor)
        if self.num_cofactor == 0:
            s += "There are no cofactors in the Network, please add cofactors first!\n"
            return s
        for idx, cofactor in self.id2cofactor.items():
            s += "ID: {}\n".format(idx)
            s += cofactor.__str__()
        # 2. print Adjacency matrix information
        if isinstance(self.D, type(None)):
            s += "------------\n"
            s += "The adjacency matrix has not been calculated yet!\n"
            s += "------------\n"
            return s
        s += "------------\n"
        s += "Adjacency matrix for the Network\n"
        s += "------------\n"
        s += self.D.__str__()
        # 3. print Reservoir information
        if self.num_reservoir == 0:
            s += "------------\n"
            s += "There are no reservoir defined in this system!\n"
            s += "------------\n"
        else:
            s += "------------\n"
            s += "There are {} reservoirs in this system.\n"
            s += "------------\n"
            for res_id, info in self.reservoirInfo.items():
                name, cofactor, redox_state, num_electron, deltaG, rate = info
                s += "------------\n"
                s += "Reservoir ID: {}, Reservoir Name: {}, connects with Cofactor ID {} with Redox State {}\n".format(res_id, name, self.cofactor2id[cofactor], redox_state)
                s += "Number of electron it exchanges at a time: {}\n".format(num_electron)
                s += "Delta G for transfering electron: {}\n".format(deltaG)
                s += "ET rate: {}\n".format(rate)
                s += "------------\n"

        return s

    def addCofactor(self, cofactor: Cofactor, sink: int):
        """
        Add cofactor into this Network
        Arguments:
            cofactor {Cofactor} -- Cofactor object
            sink : 0 or 1
            if 0, the cofactor is not a sink
            if 1, the cofactor is a sink
        """
        self.num_state *= (cofactor.capacity +1)        # The total number of possible states is equal to the product of sitecapacities+1 of each site.
                                             # (ex.) "Cofactor_1":0,1, "Cofactor_2":0,1,2 -> num_states=(cap_1+1)*(cap_2+1)=(1+1)*(2+1)=2*3=6

        if sink == 0:
            self.id2cofactor[self.num_cofactor] = cofactor   #Starts with self.num_cofactor=0, Gives an ID to cofactors that are added one by one
            self.cofactor2id[cofactor] = self.num_cofactor   #ID of the cofactor just added is basically equal to how many cofactors present in the network
            self.siteCapacity.append(cofactor.capacity)    #Trajectory of cofactor -> id -> capacity of cofactor
            self.num_cofactor += 1    #The number of cofactor counts up
        if sink == 1:
            ### Sink is treated as both a cofactor and a sink ###
            # To add a sink as a cofactor #
            self.id2cofactor[self.num_cofactor] = cofactor   #Starts with self.num_cofactor=0, Gives an ID to cofactors that are added one by one
            self.cofactor2id[cofactor] = self.num_cofactor   #ID of the cofactor just added is basically equal to how many cofactors present in the network
            self.siteCapacity.append(cofactor.capacity)    #Trajectory of cofactor -> id -> capacity of cofactor
            self.num_cofactor += 1    #The number of cofactor counts up
            # To add a sink as a sink #
            self.id2sink[self.num_sink] = cofactor
            self.sink2id[cofactor] = self.num_sink
            self.num_sink += 1

    def addConnection(self, cof1: Cofactor, cof2: Cofactor, distance: float):
        """
        "Physically" connect two cofactors in the network, allow electron to flow
        Arguments:
            cof1 {Cofactor} -- Cofactor instance
            cof2 {Cofactor} -- Cofactor instance
            distance {float} -- Distance between two cofactors, unit in angstrom
        """
        self.adjacencyList.append((self.cofactor2id[cof1], self.cofactor2id[cof2], distance))  #Append ID of cof1, ID of cof2 and distance between cof1 and cof2 to adjacency list 

    def addReservoir(self, name: str, cofactor: Cofactor, redox: int, num_electron: int, deltaG: float, rate: float):
        """
        Add an electron reservoir to the network: which cofactor it exchanges electrons with, how many electrons are exchanged at a time, the deltaG of the exchange, and the rate
        Arguments:
            name {str} -- Name of the reservoir
            cofactor {Cofactor} -- Cofactor the reservoir exchanges electrons
            redox {int} -- Redox state of the cofactor that exchanges electrons with 
            num_electron {int} -- Number of electrons exchanged at a time
            deltaG {float} -- DeltaG of the exchange
            rate {float} -- In rate
        """
        # key: (reservoir_id, cofactor_id)
        # value: list of six variables, [name, cofactor, redox_state, num_electron, deltaG, rate]
        self.id2reservoir[self.num_reservoir] = name
        self.reservoir2id[name] = self.num_reservoir
        self.reservoirInfo[self.num_reservoir] = [name, cofactor, redox, num_electron, deltaG, rate]
        self.num_reservoir += 1

    def set_Max_Electrons(self, max_electrons: int):
            self.max_electrons = max_electrons

    def set_Min_Electrons(self, min_electrons: int):
            self.min_electrons = min_electrons

    def set_sinkrate(self, sink_rate: float):
            self.sink_rate = sink_rate

    def evolve(self, t: float, pop_init: np.array) -> np.array:
        """
        Evolve the population vector with a timestep of t
        Arguments:
            t {float} -- Time
        Keyword Arguments:
            pop_init {numpy.array} -- Initial population vector (default: None)
        Returns:
            numpy.array -- Final population vector 
        """
        if pop_init is None:
            # if no pop_init is given in the input, give a default initialization
            pop_init = np.zeros(self.adj_num_state)
            # this is the initialization for 1-e case
            pop_init[0] = 1

        return linalg.expm(self.K * t) @ pop_init
    
    def evolve_new2(self, K_new: np.array, U_new: np.array, U_new_inv: np.array, t: float, pop_init: np.array) -> np.array:
        """
        Evolve the population vector with a timestep of t
        Arguments:
            K_new {mat} -- diagonalized K
            U_new {mat} -- lines up P to match new K
            t {float} -- Time
        Keyword Arguments:
            pop_init {numpy.array} -- Initial population vector (default: None)
        Returns:
            numpy.array -- Final population vector 
        """
        if pop_init is None:
            # if no pop_init is given in the input, give a default initialization
            pop_init = np.zeros(self.adj_num_state)
            # this is the initialization for 1-e case
            pop_init[0] = 1

        return U_new @ linalg.expm(K_new * t) @ U_new_inv @ pop_init
    
    def evolve_new3(self, K_eig: np.array, U_new: np.array, U_new_inv: np.array, t: float, pop_init: np.array) -> np.array:
        """
        Evolve the population vector with a timestep of t
        Arguments:
            K_eig {vec} -- K eigenvalues
            U_new {mat} -- lines up P to match new K
            t {float} -- Time
        Keyword Arguments:
            pop_init {numpy.array} -- Initial population vector (default: None)
        Returns:
            numpy.array -- Final population vector 
        """
        if pop_init is None:
            # if no pop_init is given in the input, give a default initialization
            pop_init = np.zeros(self.adj_num_state)
            # this is the initialization for 1-e case
            pop_init[0] = 1

        return U_new @ np.diag(np.exp(K_eig * t)) @ U_new_inv @ pop_init

    def ET(self, deltaG: float, R: float, reorgE, beta, V) -> float:
        """
        Calculate the nonadiabatic ET rate according to Marcus theory
        Arguments:
            deltaG {float} -- reaction free energy, unit: eV
            R {float} -- distance for decay factor, unit: angstrom
            reorgE {float} -- reorganization energy, unit: eV
            beta {float} -- inverse of kT, unit: 1/eV
            V {float} -- electronic coupling, unit: eV
        Returns:
            float -- nonadiabatic ET rate, unit: 1/s
        """
        ET_rate = (2*math.pi/self.hbar)*(self.V**2)*np.exp(-R)*(1/(math.sqrt(4*math.pi*(1/beta)*reorgE)))*np.exp(-beta*(deltaG + reorgE)**2/(4*reorgE))

        return ET_rate

    def constructAdjacencyMatrix(self):
        """
        Build adjacency matrix from the adjacency list
        """
        # obtain the dimension of this matrix
        dim = self.num_cofactor
        self.D = np.zeros((dim, dim), dtype=float)
        for item in self.adjacencyList:
            id1, id2, distance = item
            # we allow electron to flow back and forth between cofactors, thus D matrix is symmetric
            self.D[id1][id2] = self.D[id2][id1] = distance

    ####################################################
    ####  Core Functions for Building Rate Matrix   ####
    ####################################################

    # For the following functions, we make use of the internal labelling of the
    # states which uses one index which maps to the occupation number
    # representation [n1, n2, n3, ..., nN] and convert to idx in the rate
    # back and forth with state2idx() and idx2state() functions.
    
    def getCofactorEnergy(self, cof_id, redox_state):
    # Calculate the energy of the cofactor when the redox state is "redox_state".
        cof = self.id2cofactor[cof_id]
        energy = 0
        if redox_state == 0:
            energy += 0     # Energy is 0 when the cofactor is fully oxidized
        else:
            for n in range(redox_state):
                energy += -cof.redox[n]    # Energy is \sum_{x+1} (-(x+1th redox potential)) x -> x+1 

        return energy
    
    def makeEnergylist(self) -> list:
    # make energy list for all microstates assuming no interactions.
        self.Elist = []
        for i in range (self.adj_num_state):
            microstate_energy = 0
            for cof_id in range(self.num_cofactor):
                redox_state = self.idx2state(self.allow[i])[cof_id]
                microstate_energy += self.getCofactorEnergy(cof_id, redox_state)
            self.Elist.append(microstate_energy)

        return self.Elist
    
    def state2idx(self, state: list) -> int:
        """
        Given the list representation of the state, return index number in the main rate matrix
        Arguments:
            state {list} -- List representation of the state
        Returns:
            int -- Index number of the state in the main rate matrix
        """
        idx = 0
        N = 1
        for i in range(self.num_cofactor):
            idx += state[i] * N
            N *= (self.siteCapacity[i] + 1)

        return idx

    def idx2state(self, idx: int) -> list:
        """
        Given the index number of the state in the main rate matrix, return the list representation of the state
        Arguments:
            idx {int} -- Index number of the state in the main rate matrix
        Returns:
            list -- List representation of the state
        """
        state = []
        for i in range(self.num_cofactor):
            div = self.siteCapacity[i] + 1
            idx, num = divmod(idx, div)
            state.append(num)

        return state

    def constructStateList(self) -> list:
        self.allow = []
        if self.max_electrons == None and self.min_electrons == None:
            self.max_electrons = sum([site for site in self.siteCapacity])
            self.min_electrons = 0
        if self.max_electrons != None and self.min_electrons == None:
            for i in range(self.num_state):
                if sum(self.idx2state(i)) <= self.max_electrons:
                    # print(self.idx2state(i))
                    self.allow.append(i)
        if self.max_electrons == None and self.min_electrons != None:
            for i in range(self.num_state):
                if sum(self.idx2state(i)) >= self.min_electrons:
                    # print(self.idx2state(i))
                    self.allow.append(i)
        if self.max_electrons != None and self.min_electrons != None:
            for i in range(self.num_state):
                if sum(self.idx2state(i)) <= self.max_electrons and sum(self.idx2state(i)) >= self.min_electrons:
                # if sum(self.idx2state(i)) <= self.max_electrons and sum(self.idx2state(i)) >= self.min_electrons and (self.idx2state(i))[0] % 2 == 0:
                    # print(self.idx2state(i))
                    self.allow.append(i)
        self.adj_num_state = len(self.allow)

    def getRate(self, kb: float, deltaG: float):
        #rate is the rate you will input in the addReservoir
        #kb is the rate of cofactor -> reservoir
        rate = kb * np.exp(-self.beta*deltaG)
        return rate

    def connectStateRate(self, cof_i: int, red_i: int, cof_f: int, red_f: int, k: float, deltaG: float, num_electrons: int):
        """
        Add rate constant k between electron donor (cof_i) and acceptor (cof_f) with initial redox state and final redox state stated (red_i, red_f)
        ADDITION: this function combine with detailed balancing feature, helps to save initialization time.
        Arguments:
            cof_i {int} -- Donor cofactor ID
            red_i {int} -- Redox state for donor ID
            cof_f {int} -- Acceptor cofactor ID
            red_f {int} -- Redox state for acceptor ID
            k {float} -- forward state
            deltaG {float} -- deltaG between initial state and final state
        """
        for i in range(self.adj_num_state):
            # loop through all allowed states, to look for initial (donor) state
            if self.idx2state(self.allow[i])[cof_i] == red_i and self.idx2state(self.allow[i])[cof_f] == red_f:
                """
                ex. idx:some (allowed) number -> state:[0 1 1 0 2 3 ...]
                    idx2state(allow[i])[cof_i] -> ith element of the state:[0 1 1 0 2 3...]
                Basically, this "if" statement means: 
                "If cof_ith element of the state:[0 1 1 0 2 3...] is equal to the redox state of the cof_i" and also 
                "If cof_fth element of the state:[0 1 1 0 2 3...] is equal to the redox state of the cof_f"
                """
                for j in range(self.adj_num_state):
                    # loop through all allowed states, to look for final (acceptor) state
                    if self.idx2state(self.allow[j])[cof_i] == red_i - num_electrons and self.idx2state(self.allow[j])[cof_f] == red_f + num_electrons:
                        """
                        ex. idx:some allowed number -> state:[0 1 1 0 2 3 ...]
                            idx2state(allow[i])[cof_i] -> ith element of the state:[0 1 1 0 2 3...]
                        Basically, this "if" statement means: 
                        "If cof_ith element of the state:[0 1 1 0 2 3...] is equal to the (redox state - 1) (donates electron so this cofactor is oxidized) of the cof_i" and also 
                        "If cof_fth element of the state:[0 1 1 0 2 3...] is equal to the (redox state + 1) (accepts electron so this cofactor is reduced) of the cof_f"
                           """
                        # initial, final state found! check other electron conservation
                        I = np.delete(self.idx2state(self.allow[i]), [cof_i, cof_f])     # Do not allow changes other than cof_i and cof_f we have searched for
                        J = np.delete(self.idx2state(self.allow[j]), [cof_i, cof_f])     # Deleting the cof_i and cof_f that is already checked to be correct
                        if np.array_equal(I, J):   # Checking that sites other than cof_i and cof_f hasn't changed
                            # i and j state found!
                            kf = k  # forward rate
                            kb = k * np.exp(self.beta*deltaG)
                            self.K[j][i] += kf  # add population of final state, forward process
                            self.K[i][i] -= kf  # remove population of initial state, forward process   #Diagonal elements are the negative sum of the other elements in the same column
                            self.K[i][j] += kb  # add population of initial state, backward process
                            self.K[j][j] -= kb  # remove population of final sate, backward process

    def connectReservoirRate(self, cof_id: int, red_i: int, red_f: int, k: float, deltaG: float):
        """
        Add rate constant k between red_i and red_f of a cofactor, which is connected to a reservoir
        ADDITION: this function combine with detailed balancing feature, helps to save initialization time.
        Arguments:
            cof_id {int} -- Cofactor ID
            red_i {int} -- Redox state for cofactor
            red_f {int} -- Redox state for cofactor
            k {float} -- forward state
            deltaG {float} -- deltaG between initial state and final state
        """
        #if self.max_electrons == None:
        self.max_electrons = sum([site for site in self.siteCapacity])
        for i in range(self.adj_num_state):
            # loop through all allowed states, to look for initial (donor) state
            #if sum(self.idx2state(self.allow[i])) <= self.max_electrons:
             if self.idx2state(self.allow[i])[cof_id] == red_i:
                    """
                    ex. idx:some number -> state:[0 1 1 0 2 3 ...]
                    idx2state(i)[cof_i] -> ith element of the state:[0 1 1 0 2 3...]
                    Basically, this "if" statement means:
                    "If cof th element of the state:[0 1 1 0 2 3...] is equal to the redox state of the cofactor"
                    """
                    for j in range(self.adj_num_state):
                    # loop through all allowed states, to look for final (acceptor) state
                        if self.idx2state(self.allow[j])[cof_id] == red_f:
                            """
                            ex. idx:some number -> state:[0 1 1 0 2 3 ...]
                            idx2state(i)[cof_i] -> ith element of the state:[0 1 1 0 2 3...]
                            Basically, this "if" statement means: 
                            "If cof th element of the state:[0 1 1 0 2 3...] is equal to the redox state of the final cofactor"
                            """
                            # initial, final state found! check other electron conservation
                            I = np.delete(self.idx2state(self.allow[i]), [cof_id])
                            J = np.delete(self.idx2state(self.allow[j]), [cof_id])
                            if np.array_equal(I, J):
                                # i and j state found!
                                kf = k  # forward rate
                                kb = k * np.exp(self.beta*deltaG)
                                self.K[j][i] += kf  # add population of final state, forward process
                                self.K[i][i] -= kf  # remove population of initial state, forward process
                                self.K[i][j] += kb  # add population of initial state, backward process
                                self.K[j][j] -= kb  # remove population of final state, backward process

    def addMultiElectronConnection(self, cof_i, cof_f, donor_state: int, acceptor_state: int, num_electrons, k):
        i = self.cofactor2id[cof_i]
        f = self.cofactor2id[cof_f]   # Finding the name of cofactor of the ijth of the adjacency matrix
        deltaG = sum([cof_i.redox[donor_state-num_electrons + n] - cof_f.redox[acceptor_state+n] for n in range(0, num_electrons)])
        self.connectStateRate(i, donor_state, f, acceptor_state, k, deltaG, num_electrons)   #Adding the rate constant to rate matrix

    def constructRateMatrix(self):
        """
        Build rate matrix
        """
        # initialize the rate matrix with proper dimension
        self.K = np.zeros((self.adj_num_state, self.adj_num_state), dtype=float)      #The dimension of the rate matrix is basically equal to the total number of states
        # microstate_energy = self.microstate_energy()
        # loop through cofactor_id in adjacency matrix
        """
        Take the adjacency matrix which is weighted by the distance to construct the full rate matrix
        """
        sink_list = []
        for sink_id in range(self.num_sink):
            sink_list.append(self.id2sink[sink_id])
        ##########################
        ## cofactor-cofactor ET ##
        ##########################
        for donor_id in range(self.num_cofactor):
            donor = self.id2cofactor[donor_id]
            #for acceptor_id in range(self.num_cofactor):
            for acceptor_id in range(donor_id+1, self.num_cofactor):    # These two "for" loops take care of (upper triangular - diagonal) part of the adjacency matrix
                acceptor = self.id2cofactor[acceptor_id]
                if self.D[donor_id][acceptor_id] != 0:  # Find cofactors that are physically connected
                    dis = self.D[donor_id][acceptor_id]
                    for donor_state in range(1, donor.capacity+1):   # If the donor is in any redox state such that it has an electron available to donate
                        for acceptor_state in range(0, acceptor.capacity):   # If the acceptor is in any redox state such that it has capacity to accept an electron
                            deltaG = donor.redox[donor_state-1] - acceptor.redox[acceptor_state]
                            ## Loop through allowed microstates to find microstate-microstate transitions that correspond to the cofactor-cofactor ET found ##
                            for initial in range(self.adj_num_state):
                                for final in range(self.adj_num_state):
                                    # Find the initial microstate
                                    if self.idx2state(self.allow[initial])[donor_id] == donor_state and self.idx2state(self.allow[initial])[acceptor_id] == acceptor_state:
                                        # Find the final microstate
                                        if self.idx2state(self.allow[final])[donor_id] == donor_state-1 and self.idx2state(self.allow[final])[acceptor_id] == acceptor_state+1:
                                            # Microstates found! Check for other electron conservation
                                            I = np.delete(self.idx2state(self.allow[initial]), [donor_id, acceptor_id])
                                            J = np.delete(self.idx2state(self.allow[final]), [donor_id, acceptor_id])
                                            if np.array_equal(I, J):
                                                if acceptor not in sink_list:   # If the acceptor is not a sink
                                                    kf = self.ET(deltaG, dis, self.reorgE, self.beta, self.V)
                                                    kb = kf * np.exp(self.beta*deltaG)
                                                    self.K[final][initial] = kf
                                                    self.K[initial][initial] -= kf
                                                    self.K[initial][final] = kb
                                                    self.K[final][final] -= kb
                                                    # print("Cofactor-cofactor ET")
                                                    # print([donor.name, acceptor.name], ":", [donor_state, acceptor_state], "->", [donor.name, acceptor.name], ":", [donor_state-1, acceptor_state+1])
                                                    # print(self.idx2state(self.allow[initial]), self.idx2state(self.allow[final]), self.K[final][initial], self.K[initial][final])
                                                ######################
                                                ## Cofactor-Sink ET ##
                                                ######################
                                                else:   # If the acceptor is a sink
                                                    kf_sink = self.sink_rate   # Electron transfer rate into a sink
                                                    kb_sink = 0  # The rate to go out from the sink is 0
                                                    self.K[final][initial] = kf_sink
                                                    self.K[initial][initial] -= kf_sink
                                                    self.K[initial][final] = kb_sink
                                                    self.K[final][final] -= kb_sink
                                                    # print("Cofactor-sink ET")
                                                    # print([donor.name, acceptor.name], ":", [donor_state, acceptor_state], "->", [donor.name, acceptor.name], ":", [donor_state-1, acceptor_state+1])
                                                    # print(initial,":",self.idx2state(self.allow[initial]), final,":",self.idx2state(self.allow[final]), self.K[final][initial], self.K[initial][final])

    def two_electron_reservoir_step(self, DR: Cofactor, D: Cofactor, reservoir_rate):
        DR_id = self.cofactor2id[DR]
        D_id = self.cofactor2id[D]
        dis = self.D[DR_id][D_id]
        for initial in range(self.adj_num_state):
            for final in range(self.adj_num_state):
                # initial:[DR, D, ...] -> final:[DR+1, D-1, ...]
                if self.idx2state(self.allow[final])[DR_id]-self.idx2state(self.allow[initial])[DR_id] == 1:
                    if self.idx2state(self.allow[initial])[D_id]-self.idx2state(self.allow[final])[D_id] == 1:
                        # Microstates found! Check for other electron conservation
                        I = np.delete(self.idx2state(self.allow[initial]), [DR_id, D_id])
                        J = np.delete(self.idx2state(self.allow[final]), [DR_id, D_id])
                        if np.array_equal(I, J):
                            ## forward: cofactor -> reservoir
                            original_kf = self.K[final][initial]
                            original_kb = self.K[initial][final]
                            # Account for changes in the diagonal elements
                            self.K[initial][initial] += original_kf
                            self.K[final][final] += original_kb
                            # Reconstruct rate matrix
                            self.K[final][initial] = 0
                            self.K[initial][initial] -= 0
                            self.K[initial][final] = 0        # All one-electron transfer from DR to D is set to zero
                            self.K[final][final] -= 0      # Account for changes in the diagonal elements
                            # print(self.idx2state(self.allow[initial]), self.idx2state(self.allow[final]), self.K[final][initial], self.K[initial][final])
                # initial:[DR, D, ...] -> final:[DR+2, D-2, ...]
                if self.idx2state(self.allow[final])[DR_id]-self.idx2state(self.allow[initial])[DR_id] == 2:
                    if self.idx2state(self.allow[initial])[D_id]-self.idx2state(self.allow[final])[D_id] == 2:
                        # Microstates found! Check for other electron conservation
                        I = np.delete(self.idx2state(self.allow[initial]), [DR_id, D_id])
                        J = np.delete(self.idx2state(self.allow[final]), [DR_id, D_id])
                        if np.array_equal(I, J):
                            ## forward: cofactor -> reservoir
                            # All two-electron transfer from DR to D is given a rate constant
                            deltaG = 2*((D.redox[0]+D.redox[1])/2 - DR.redox[0])    #DeltaG for cofactor -> reservoir
                            # deltaG_microstate = self.Elist[final] - self.Elist[initial]
                            # print(deltaG, deltaG_microstate)
                            kf = self.ET(deltaG, dis, self.reorgE, self.beta, self.V)
                            # kf = reservoir_rate     # D -> DR rate (cofactor -> reservoir rate)
                            kb = kf * np.exp(self.beta*deltaG)    # DR -> D rate (reservoir -> cofactor rate)
                            original_kf = self.K[final][initial]
                            original_kb = self.K[initial][final]
                            # Account for changes in the diagonal elements
                            self.K[initial][initial] += original_kf
                            self.K[final][final] += original_kb
                            # Reconstruct rate matrix
                            self.K[final][initial] = kf
                            self.K[initial][initial] -= kf
                            self.K[initial][final] = kb
                            self.K[final][final] -= kb
                            # print("forward:", kf, "reverse:", kb)
                            # print(self.idx2state(self.allow[initial]), self.idx2state(self.allow[final]), self.K[final][initial], self.K[initial][final], deltaG)

    def two_electron_reservoir_step_notconcerted(self, DR: Cofactor, D: Cofactor):
        DR_id = self.cofactor2id[DR]
        D_id = self.cofactor2id[D]
        for initial in range(self.adj_num_state):
            for final in range(self.adj_num_state):
                # initial:[DR, D, ...] -> final:[DR+1, D-1, ...]
                if self.idx2state(self.allow[final])[DR_id]-self.idx2state(self.allow[initial])[DR_id] == 1:
                    if self.idx2state(self.allow[initial])[D_id]-self.idx2state(self.allow[final])[D_id] == 1:
                        # Microstates found! Check for other electron conservation
                        I = np.delete(self.idx2state(self.allow[initial]), [DR_id, D_id])
                        J = np.delete(self.idx2state(self.allow[final]), [DR_id, D_id])
                        if np.array_equal(I, J):
                            ## forward: cofactor -> reservoir
                            original_kf = self.K[final][initial]
                            original_kb = self.K[initial][final]
                            # Account for changes in the diagonal elements
                            self.K[initial][initial] += original_kf
                            self.K[final][final] += original_kb
                            # Reconstruct rate matrix
                            dis = self.D[DR_id][D_id]
                            redox_state_DR = self.idx2state(self.allow[initial])[DR_id]
                            redox_state_D = self.idx2state(self.allow[final])[D_id]
                            deltaG = DR.redox[redox_state_DR-1] - D.redox[redox_state_D-1]
                            new_kf = self.ET(deltaG, dis, self.reorgE, self.beta, self.V)
                            new_kb = new_kf * np.exp(self.beta*deltaG)
                            self.K[final][initial] = new_kf
                            self.K[initial][initial] -= new_kf
                            self.K[initial][final] = new_kb
                            self.K[final][final] -= new_kb
                            # print(self.idx2state(self.allow[initial]), self.idx2state(self.allow[final]), self.K[final][initial], self.K[initial][final])

    def one_electron_reservoir_step(self, reservoir: Cofactor, cofactor: Cofactor, reservoir_rate):
        cof_id = self.cofactor2id[cofactor]
        res_id = self.cofactor2id[reservoir]
        for initial in range(self.adj_num_state):
            for final in range(self.adj_num_state):
                if self.K[final][initial] != 0:
                    # initial:[..., L2, LR] -> final:[..., L2-1, LR+1] 
                    if self.idx2state(self.allow[initial])[cof_id]-self.idx2state(self.allow[final])[cof_id] == 1:   
                        if self.idx2state(self.allow[final])[res_id]-self.idx2state(self.allow[initial])[res_id] == 1:  
                            ## forward: cofactor -> reservoir
                            deltaG = cofactor.redox[0] - reservoir.redox[0]
                            # deltaG = self.Elist[final] - self.Elist[initial]
                            kf = reservoir_rate     # L2(H2) -> LR(HR) rate (cofactor -> reservoir rate)
                            kb = kf * np.exp(self.beta*deltaG)
                            original_kf = self.K[final][initial]
                            original_kb = self.K[initial][final]
                            # Account for changes in the diagonal elements
                            self.K[initial][initial] += original_kf
                            self.K[final][final] += original_kb
                            # Reconstruct rate matrix
                            self.K[final][initial] = kf
                            self.K[initial][initial] -= kf
                            self.K[initial][final] = kb
                            self.K[final][final] -= kb
                            # print(self.idx2state(self.allow[initial]), self.idx2state(self.allow[final]), self.K[final][initial], self.K[initial][final], deltaG)

    def one_electron_reservoir_step_NAD(self, reservoir: Cofactor, cofactor: Cofactor, reservoir_rate):
        cof_id = self.cofactor2id[cofactor]
        res_id = self.cofactor2id[reservoir]
        for initial in range(self.adj_num_state):
            for final in range(self.adj_num_state):
                if self.K[final][initial] != 0:
                    # initial:[..., L2(2), LR] -> final:[..., L2-1(1), LR+1] 
                    if self.idx2state(self.allow[initial])[cof_id]-self.idx2state(self.allow[final])[cof_id] == 1 and self.idx2state(self.allow[initial])[cof_id] == 2:   
                        if self.idx2state(self.allow[final])[res_id]-self.idx2state(self.allow[initial])[res_id] == 1:  
                            ## forward: cofactor -> reservoir
                            deltaG = cofactor.redox[1] - reservoir.redox[0]
                            # deltaG = self.Elist[final] - self.Elist[initial]
                            kf = reservoir_rate     # L2(H2) -> LR(HR) rate (cofactor -> reservoir rate)
                            kb = kf * np.exp(self.beta*deltaG)
                            original_kf = self.K[final][initial]
                            original_kb = self.K[initial][final]
                            # Account for changes in the diagonal elements
                            self.K[initial][initial] += original_kf
                            self.K[final][final] += original_kb
                            # Reconstruct rate matrix
                            self.K[final][initial] = kf
                            self.K[initial][initial] -= kf
                            self.K[initial][final] = kb
                            self.K[final][final] -= kb
                            # print(self.idx2state(self.allow[initial]), self.idx2state(self.allow[final]), self.K[final][initial], self.K[initial][final], deltaG)
                    # initial:[..., L2(1), LR] -> final:[..., L2-1(0), LR+1]
                    if self.idx2state(self.allow[initial])[cof_id]-self.idx2state(self.allow[final])[cof_id] == 1 and self.idx2state(self.allow[initial])[cof_id] == 1:   
                        if self.idx2state(self.allow[final])[res_id]-self.idx2state(self.allow[initial])[res_id] == 1:  
                            ## forward: cofactor -> reservoir
                            deltaG = cofactor.redox[0] - reservoir.redox[0]
                            # deltaG = self.Elist[final] - self.Elist[initial]
                            kf = reservoir_rate     # L2(H2) -> LR(HR) rate (cofactor -> reservoir rate)
                            kb = kf * np.exp(self.beta*deltaG)
                            original_kf = self.K[final][initial]
                            original_kb = self.K[initial][final]
                            # Account for changes in the diagonal elements
                            self.K[initial][initial] += original_kf
                            self.K[final][final] += original_kb
                            # Reconstruct rate matrix
                            self.K[final][initial] = kf
                            self.K[initial][initial] -= kf
                            self.K[initial][final] = kb
                            self.K[final][final] -= kb
                            # print(self.idx2state(self.allow[initial]), self.idx2state(self.allow[final]), self.K[final][initial], self.K[initial][final], deltaG)

    ########################################
    ####    Data Analysis Functions     ####
    ########################################

    def population(self, pop: np.array, cofactor: Cofactor, redox: int) -> float:
        """
        Calculate the population of a cofactor in a given redox state
         -> (ex.)pop=[1 0 0 2 5 ...]:len(pop)=num_state, pop is the population vector of the states. (pop[0]=population of state[0], pop[1]=population of state[1]...)

        Arguments:
            pop {numpy.array} -- Population vector     This is the population vector of the states. len(pop)=self.adj_num_state
            cofactor {Cofactor} -- Cofactor object
            redox {int} -- Redox state of the cofactor
        Returns:
            float -- Population of the cofactor at specific redox state
        """
        cof_id = self.cofactor2id[cofactor]
        ans = 0
        for i in range(len(pop)):
            #Loop through all the possible states
            if self.idx2state(self.allow[i])[cof_id] == redox:   #For every state, the number of electrons on each site is known, (ex.)state[0]=[1 2 0 3 2...], state[1]=[0 2 3 1 ...]
                # It loops through all the states to find where the cof th element of (ex.)state:[0 1 1 0 2 3...] is equal to the given redox state
                # Population of electron at each cofactor = redox state of that cofactor
                ans += pop[i]

        return ans

    def getCofactorRate(self, cof_i: Cofactor, red_i: int, cof_f: Cofactor, red_f: int, pop: np.array) -> float:
        """
        Calculate the instantaneous forward rate from cof_i to cof_f
        Arguments:
            cof_i {Cofactor} -- Cofactor object for initial cofactor
            red_i {int} -- Redox state for initial cofactor
            cof_f {Cofactor} -- Cofactor object for final cofactor
            red_f {int} -- Redox state for final cofactor
            pop {np.array} -- Population vector      This is the population vector of the states. len(pop)=self.num_state
        Returns:
            float -- Instantaneous forward rate
        """
        cof_i_id = self.cofactor2id[cof_i]
        cof_f_id = self.cofactor2id[cof_f]
        flux = 0
        for i in range(self.adj_num_state):
            # loop through all states, to find initial state
            if self.idx2state(self.allow[i])[cof_i_id] == red_i and self.idx2state(self.allow[i])[cof_f_id] == red_f - 1:
                """
                This "if" statement means: 
                "If the element that corresponds to cof_i in the state:[0 1 1 0 2 3...] is equal to the redox state of cof_i (prior to donating an electron)" and
                "If the element that corresponds to cof_f in the state:[0 1 1 0 2 3...] is equal to the (redox state of cof_f -1) (prior to accepting an electron)"
                """
                for j in range(self.adj_num_state):
                    # loop through all states, to find final state
                    if self.idx2state(self.allow[j])[cof_i_id] == red_i - 1 and self.idx2state(self.allow[j])[cof_f_id] == red_f:
                        """
                        This "if" statement means: 
                        "If the element that corresponds to cof_i in the state:[0 1 1 0 2 3...] is equal to the (redox state of cof_i -1) (donated an electron)" and
                        "If the element that corresponds to cof_f in the state:[0 1 1 0 2 3...] is equal to the redox state of cof_f (accepted an electron)"
                        """
                        # initial, final state found! check other electron conservation
                        I = np.delete(self.idx2state(self.allow[i]), [cof_i_id, cof_f_id])
                        J = np.delete(self.idx2state(self.allow[j]), [cof_i_id, cof_f_id])
                        if np.array_equal(I, J):
                            # i and j state found!)
                            flux += self.K[j][i] * pop[i]      #K is rate matrix, so len(K)=self.num_state

        return flux

    def getCofactorFlux(self, cof_i: Cofactor, red_i: int, cof_f: Cofactor, red_f: int, pop: np.array) -> float:
        """
        Calculate the instantaneous NET flux from initial cofactor(state) to final cofactor(state), by calling getCofactorRate() twice
        Arguments:
            cof_i {Cofactor} -- Cofactor object for initial cofactor
            red_i {int} -- Redox state for initial cofactor before ET
            cof_f {Cofactor} -- Cofactor object for final cofactor
            red_f {int} -- Redox state for final cofactor after ET
            pop {np.array} -- Population vector      This is the population vector of the states. len(pop)=self.num_state
        Returns:
            float -- Instantaneous net flux
        """
        return self.getCofactorRate(cof_i, red_i, cof_f, red_f, pop) - self.getCofactorRate(cof_f, red_f, cof_i, red_i, pop)

    # def getReservoirFlux(self, name: str, pop: np.array) -> float:
    #     """
    #     Calculate the instantaneous net flux into the reservoir connected to the reservoir
    #     Arguments:
    #         reservoir_id {int} -- Reservoir ID
    #         pop {np.array} -- Population vector      This is the population vector of the states. len(pop)=self.num_state
    #     Returns:
    #         float -- Instantaneous net flux connected to the reservoir
    #     """
    #     reservoir_id = self.reservoir2id[name]
    #     name, cofactor, redox_state, num_electron, deltaG, rate=self.reservoirInfo[reservoir_id]
    #     reverse_rate = rate * np.exp(self.beta*deltaG)
    #     final_redox=redox_state-num_electron      #redox_state is basically the initial redox state of the cofactor, which is info stored in reservoirInfo dict()

    #     return (self.population(pop, cofactor, redox_state) * rate - self.population(pop, cofactor, final_redox) * reverse_rate) * num_electron

    def getExptvalue(self, pop: np.array, cofactor: Cofactor) -> float:
        """
        Calculate the expectation value of the number of particles at a given cofactor at a given time
        Arguments:
            cofactor {Cofactor} -- Cofactor object
            pop {cp.array} -- Population vector of the states
        """
        cof_id = self.cofactor2id[cofactor]
        expt=0
        #loop through all the possible states
        for i in range(self.adj_num_state):
            expt+=self.idx2state(self.allow[i])[cof_id]*pop[i]   #sum((number of particle)*(probability))

        return expt
    
    def popState(self, pop_init: np.array, t: float) -> list:
        """
        Visualize the population of the microstates at a given time
        Arguments:
            t {float} -- given time
            pop_init {numpy.array} -- Initial population vector (default: None)
        Returns:
            list -- [[population, microstate that corresponds to that population]]
        """
        popstate=[]
        pop=self.evolve(t, pop_init)
        for i in range(self.adj_num_state):
            popstate.append([pop[i], self.idx2state(self.allow[i])])

        return popstate

    def getJointExptvalue(self, pop: np.array, cofactor_1: Cofactor, red_1: int, cofactor_2: Cofactor, red_2: int) -> float:
        """
        Calculate the joint probability of cofactor_1 being in redox state (red_1) and cofactor_2 being in redox state (red_2)
        Arguments:
            cofactor {Cofactor} -- Cofactor object
            pop {cp.array} -- Population vector of the states
        """
        cof1_id = self.cofactor2id[cofactor_1]
        cof2_id = self.cofactor2id[cofactor_2]
        expt=0
        for i in range(self.adj_num_state):
            if self.idx2state(self.allow[i])[cof1_id] == red_1 and self.idx2state(self.allow[i])[cof2_id] == red_2:
                expt += pop[i]
        return expt
    
    def getEnergyvalue(self, pop:np.array) -> float:
        Evalue = 0.0
        for i in range(self.adj_num_state):
            Evalue += pop[i]*self.Elist[i]
        return Evalue
    
    
