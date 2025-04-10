import time
import numpy as np
import streamlit as st
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

class QuantumPathfinder:
    """Container class for quantum pathfinding operations to prevent circular imports"""
    
    @staticmethod
    def grover_oracle(n: int, marked_state: int) -> QuantumCircuit:
        """
        Creates a quantum oracle that marks the optimal path
        
        Args:
            n: Number of qubits
            marked_state: The state representing the optimal path
            
        Returns:
            QuantumCircuit: Oracle gate that marks the solution state
        """
        oracle = QuantumCircuit(n)
        binary_marked = format(marked_state, f'0{n}b')

        # Flip qubits where bit is '0'
        for i, bit in enumerate(binary_marked):
            if bit == '0':
                oracle.x(i)  

        # Multi-controlled Z gate
        oracle.h(n - 1)
        oracle.mcx(list(range(n - 1)), n - 1)  
        oracle.h(n - 1)

        # Flip back
        for i, bit in enumerate(binary_marked):
            if bit == '0':
                oracle.x(i)  

        return oracle.to_gate(label="Oracle")

    @staticmethod
    def diffuser(n: int) -> QuantumCircuit:
        """
        Creates the Grover diffusion operator
        
        Args:
            n: Number of qubits
            
        Returns:
            QuantumCircuit: Diffusion gate for amplitude amplification
        """
        diffuser_circuit = QuantumCircuit(n)
        diffuser_circuit.h(range(n))
        diffuser_circuit.x(range(n))
        diffuser_circuit.h(n - 1)
        diffuser_circuit.mcx(list(range(n - 1)), n - 1)
        diffuser_circuit.h(n - 1)
        diffuser_circuit.x(range(n))
        diffuser_circuit.h(range(n))
        return diffuser_circuit.to_gate(label="Diffuser")

    @staticmethod
    def grover_search(n: int, marked_state: int, shots: int = 1024) -> dict:
        """
        Executes Grover's Algorithm to find the optimal path
        
        Args:
            n: Number of qubits
            marked_state: Target state representing optimal path
            shots: Number of quantum measurements
            
        Returns:
            dict: Measurement counts of quantum states
        """
        qc = QuantumCircuit(n, n)
        qc.h(range(n))  

        # Calculate optimal number of iterations
        num_iterations = int(np.pi / 4 * np.sqrt(2**n))  

        # Create quantum gates
        oracle = QuantumPathfinder.grover_oracle(n, marked_state)
        diffuser_gate = QuantumPathfinder.diffuser(n)

        # Apply Grover iterations
        for _ in range(num_iterations):
            qc.append(oracle, range(n))
            qc.append(diffuser_gate, range(n))

        # Measure all qubits
        qc.measure(range(n), range(n))

        try:
            simulator = AerSimulator()
            transpiled_qc = transpile(qc, simulator)
            job = simulator.run(transpiled_qc, shots=shots)
            return job.result().get_counts()
        except Exception as e:
            st.error(f"❌ Quantum Simulation Error: {e}")
            return {}

def calculate_efficiency_accuracy(actual_cost: float, 
                                computed_cost: float, 
                                execution_time: float) -> tuple:
    """
    Computes algorithm performance metrics
    
    Args:
        actual_cost: Optimal path cost from Dijkstra's
        computed_cost: Cost from evaluated algorithm
        execution_time: Runtime in seconds
        
    Returns:
        tuple: (efficiency, accuracy) metrics
    """
    efficiency = 1 / execution_time if execution_time > 0 else 0  
    accuracy = (1 - abs(actual_cost - computed_cost) / actual_cost) * 100 if actual_cost > 0 else 0  
    return round(efficiency, 4), round(accuracy, 2)

def compare_algorithms(graph: dict, 
                      start: str, 
                      end: str, 
                      n_qubits: int, 
                      marked_state: int) -> dict:
    """
    Compares classical and quantum pathfinding algorithms
    
    Args:
        graph: Network graph representation
        start: Starting node
        end: Target node
        n_qubits: Number of qubits needed
        marked_state: Quantum state representing solution
        
    Returns:
        dict: Performance metrics for all algorithms
    """
    # Lazy imports to prevent circular dependencies
    from algorithms.dijkstra import dijkstra
    from algorithms.a_star import a_star
    from algorithms.brute_force import brute_force_search

    results = {}

    # Classical Dijkstra
    start_time = time.time()
    dijkstra_path, dijkstra_cost = dijkstra(graph, start, end)
    dijkstra_time = time.time() - start_time
    dijkstra_eff, dijkstra_acc = calculate_efficiency_accuracy(
        dijkstra_cost, dijkstra_cost, dijkstra_time)
    results["Dijkstra"] = {
        "Path": dijkstra_path,
        "Cost": dijkstra_cost,
        "Time": dijkstra_time,
        "Efficiency": dijkstra_eff,
        "Accuracy": dijkstra_acc
    }

    # Classical A*
    start_time = time.time()
    a_star_path, a_star_cost = a_star(graph, start, end)
    a_star_time = time.time() - start_time
    a_star_eff, a_star_acc = calculate_efficiency_accuracy(
        dijkstra_cost, a_star_cost, a_star_time)
    results["A*"] = {
        "Path": a_star_path,
        "Cost": a_star_cost,
        "Time": a_star_time, 
        "Efficiency": a_star_eff,
        "Accuracy": a_star_acc
    }

    # Classical Brute Force
    start_time = time.time()
    brute_force_path, brute_force_cost = brute_force_search(graph, start, end)
    brute_force_time = time.time() - start_time
    brute_eff, brute_acc = calculate_efficiency_accuracy(
        dijkstra_cost, brute_force_cost, brute_force_time)
    results["Brute Force"] = {
        "Path": brute_force_path,
        "Cost": brute_force_cost,
        "Time": brute_force_time,
        "Efficiency": brute_eff,
        "Accuracy": brute_acc
    }

    # Quantum Grover's Algorithm
    start_time = time.time()
    grover_result = QuantumPathfinder.grover_search(n_qubits, marked_state)
    grover_time = time.time() - start_time
    correct_path = format(marked_state, f'0{n_qubits}b')
    correct_count = grover_result.get(correct_path, 0)
    total_shots = sum(grover_result.values())
    
    results["Grover"] = {
        "Counts": grover_result,
        "Time": grover_time,
        "Efficiency": 1 / grover_time if grover_time > 0 else 0,
        "Accuracy": (correct_count / total_shots) * 100 if total_shots > 0 else 0
    }

    return results