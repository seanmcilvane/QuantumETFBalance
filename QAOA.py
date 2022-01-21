import numpy as np
from qiskit import Aer, QuantumCircuit, ClassicalRegister, QuantumRegister, execute, assemble, transpile
from qiskit.algorithms import NumPyMinimumEigensolver
from qiskit.algorithms.optimizers import *
from qiskit_finance.data_providers import YahooDataProvider
from qiskit_finance.applications.optimization import PortfolioOptimization
from qiskit_optimization.algorithms import MinimumEigenOptimizer 
from qiskit_optimization.applications import OptimizationApplication
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit.utils import algorithm_globals


import datetime
from sympy.physics.quantum import TensorProduct
from sympy.physics.quantum.dagger import Dagger
import matplotlib.pyplot as plt

class Portfolio_Optimization:
    
    
    def __init__(self, Stocks, optimizer, backend, q = .5, layers = 4, shots = 1024):
        
        self.Stocks = Stocks
        self.optimizer = optimizer
        self.backend = backend
        self.q = q 
        self.layers = layers
        self.shots = shots
        self.n = len(Stocks)
        self.params = np.random.rand(2*self.layers) 
        self.lcost = []
        self.bchoice = []
      
        data = YahooDataProvider(tickers=Stocks, start=datetime.datetime(2021,10,1), end=datetime.datetime(2021,12,3))
        data.run()

        self.mu = data.get_period_return_mean_vector()
        self.sigma = data.get_period_return_covariance_matrix()

        
    def layer_circuit(self):
  
        qr = QuantumRegister(self.n)
        cr = ClassicalRegister(self.n)
        qc = QuantumCircuit(qr, cr) 
    
        # State preparation. Apply hadamard to all qubits to achieve equal superposition |+>_N
        for i in range(self.n):
            qc.h(qr[i])
    
        # Loops for multiple layers 
        for i in range(self.layers):
        
            # Define parameters for each layer 
            gamma = self.params[2*i]
            beta = self.params[(2*i)+1]
  
            #########################
            # Apply Cost Hamiltonian#
            #########################
        
            # Applies the expected return part of the cost Hamiltonian
            for qubit in range(self.n):
                qc.rz(-2*gamma*self.mu[qubit], qr[qubit])
        
            for qubit in range(self.n): 
            
                if qubit < self.n-1:
                    for j in range(qubit, self.n-1):
      
                    
                        qc.barrier()
                        # Swap gates to allow QAOA to run with only nearest qubits having connectivity 
                        for k in range(j-qubit):
                            qc.swap(qr[j-k], qr[j-k+1])
                    
                        qc.barrier()
    
                        # Apply ZZ interaction, the risk part of the cost Hamiltonian
                        qc.cx(qr[qubit], qr[qubit+1])
                        qc.rz(2*gamma*self.sigma[qubit][j], qr[qubit+1])
                        qc.cx(qr[qubit], qr[qubit+1]) 
                    
                        qc.barrier()
                    
                        # Return qubits to correct order
                        for k in range(j-qubit):
                            qc.swap(qr[qubit+1+k],qr[qubit+2+k])
    
                    
            ############################
            # Apply Mixing  Hamiltonian#
            ############################
        
            for qubit in range(self.n):
                qc.rx(2*beta, qr[qubit])
        
            qc.barrier()
    
        # Measure the state 
        qc.measure(qr, cr)
    
        return qc


    
    def objective_function(self, params):
        # Define the quantum circuit with updated parameters
        qc = self.layer_circuit()
        t_qc = transpile(qc, self.backend)
        qobj = assemble(t_qc, shots = self.shots)
        result = self.backend.run(qobj).result()
    
        states = result.get_counts(qc)

        max_key = int(max(states, key=states.get))  # Obtains the state with the highest counts
    
        max_string = str(max_key)                      # Turns it into string and corrects the truncation of 0's by Qiskit:(Qiskit will truncate a '010' into a '10' and this fixes that)
        if len(max_string) < self.n: 
            while len(max_string) < self.n:
                max_string = "0" + max_string
   
    # Turns the corrected string into a vector describing the choice of assets
    
        choices = np.array([int(x) for x in max_string])
        print("Choices: ", choices)

    # Calculating expectation value of the cost Hamiltonian on the trail state
        avg = 0
        sum_count = 0
    
        for bitstring, count in states.items():
        
            string = str(bitstring)                      # Turns it into string and corrects the truncation of 0's by Qiskit: (Qiskit will truncate a '010' into a '10' and this fixes that)
            if len(string) < self.n: 
                while len(string) < self.n:
                    string = "0" + string
            choice = np.array([int(x) for x in string])
        
        
            cost = self.q*np.dot(np.dot(choice, self.sigma), Dagger(choice)) - np.dot(self.mu, Dagger(choice))
            avg += cost * count
            sum_count += count
    
        expectation_value = avg/sum_count
    
        # Keep track of the most measured state and its coresponding cost
        self.bchoice.append(choices)
        low_cost = self.q*np.dot(np.dot(choices, self.sigma), Dagger(choices)) - np.dot(self.mu, Dagger(choices))
        self.lcost.append(low_cost)
    
        # Print to see the progress of the QAOA 
        print("Expectation Value: ", expectation_value)
        print("Cost: ", low_cost)
        print("")
    
    
        # Return the expectation value 
        return expectation_value

    
    
    def QAOA(self):
        
        ret = self.optimizer.optimize(num_vars = len(self.params), objective_function = self.objective_function, initial_point = self.params)
        #qc = self.layer_circuit(ret[0], self.n, self.layers)
        self.params= ret[0]
        qc = self.layer_circuit()
        t_qc = transpile(qc, self.backend)
        qobj = assemble(t_qc, shots = self.shots)

        # Find the lowest cost found and the associated choices from all iterations of QAOA.
        
        lowest_cost = min(self.lcost)
        best_choice = self.bchoice[self.lcost.index(lowest_cost)]
        
        print("Best Choice: ", best_choice)
        print("Lowest Cost: ", lowest_cost)
        
        
        return lowest_cost, best_choice

    
    def classical_solution(self): 
        
        bounds = [(0,1) for x in self.Stocks]
        portfolio = PortfolioOptimization(self.mu, self.sigma, self.q, budget = self.n, bounds = bounds)
        qp = portfolio.to_quadratic_program()
        qp.remove_linear_constraint(0)
        print(qp)
    
        optimizer = SLSQP(maxiter=1000)  
        algorithm_globals.random_seed = 1234
        backend = Aer.get_backend('qasm_simulator')
        exact_mes = NumPyMinimumEigensolver()
        exact_eigensolver = MinimumEigenOptimizer(exact_mes)
        result = exact_eigensolver.solve(qp)

        return result 
    
    
    def plot_covariance_matrix(self):
        
        fig, ax = plt.subplots(1,1)
        im = plt.imshow(self.sigma, extent = [-1,1,-1,1])
        x_label_list = reversed(self.Stocks)
        y_label_list = reversed(self.Stocks)
       
        xticks = list(np.linspace(-1+(1/self.n), 1-(1/self.n), self.n))
        yticks = list(np.linspace(1-1/self.n, -1+ 1/self.n, self.n))
        
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        
        ax.set_xticklabels(x_label_list)
        ax.set_yticklabels(y_label_list)
        plt.colorbar()
        plt.clim(-0.000002, 0.00001)
        plt.show()
        