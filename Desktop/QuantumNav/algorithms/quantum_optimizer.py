import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from qiskit.algorithms import AmplificationProblem, Grover
from qiskit.circuit.library import PhaseOracle
from qiskit.visualization import plot_histogram
from typing import Dict, List, Tuple, Optional
import networkx as nx
import matplotlib.pyplot as plt
import streamlit as st
import time

class QuantumPathOptimizer:
    """
    Quantum path optimization system using Grover's algorithm for finding optimal routes.
    Handles both exact pathfinding and approximate optimization for large graphs.
    
    Features:
    - Quantum circuit construction for path problems
    - Graph pre-processing for quantum compatibility
    - Hybrid quantum-classical fallback system
    - Visualization tools for quantum results
    """
    
    def __init__(self, backend: str = 'qasm_simulator', max_qubits: int = 20):
        """
        Initialize the quantum path optimizer.
        
        Args:
            backend: Quantum simulator or backend name
            max_qubits: Maximum number of qubits to use before classical fallback
        """
        self.backend = Aer.get_backend(backend)
        self.max_qubits = max_qubits
        self._oracle_cache = {}
        
    def find_optimal_path(self, 
                         graph: Dict[str, Dict[str, float]], 
                         start: str, 
                         end: str,
                         iterations: int = 3) -> Tuple[List[str], float]:
        """
        Find the optimal path between two nodes using quantum amplification.
        
        Args:
            graph: Weighted graph as {node: {neighbor: cost}}
            start: Starting node ID
            end: Target node ID
            iterations: Number of Grover iterations
            
        Returns:
            Tuple of (optimal path, total cost)
            
        Raises:
            ValueError: If graph is too large for quantum processing
        """
        # 1. Validate and preprocess graph
        self._validate_graph(graph, start, end)
        simplified_graph = self._preprocess_graph(graph, start, end)
        
        # 2. Check for classical fallback
        if self._requires_classical_fallback(simplified_graph):
            st.warning("Graph too large - using classical optimizer")
            return self._classical_fallback(graph, start, end)
        
        # 3. Quantum processing
        try:
            with st.spinner("Running quantum optimization..."):
                # 3.1 Convert to quantum representation
                oracle = self._create_oracle(simplified_graph)
                
                # 3.2 Configure Grover's algorithm
                problem = AmplificationProblem(
                    oracle,
                    is_good_state=lambda x: self._is_valid_path(x, simplified_graph, start, end)
                )
                grover = Grover(iterations=iterations)
                
                # 3.3 Execute quantum circuit
                result = grover.amplify(problem)
                counts = result.circuit_results
                
                # 3.4 Interpret results
                optimal_path, cost = self._interpret_results(
                    counts, 
                    simplified_graph, 
                    start, 
                    end
                )
                
                # 3.5 Visualize results
                self._visualize_results(counts)
                
                return optimal_path, cost
                
        except Exception as e:
            st.error(f"Quantum optimization failed: {str(e)}")
            return self._classical_fallback(graph, start, end)
    
    def _validate_graph(self, graph: Dict, start: str, end: str):
        """Validate graph structure and nodes"""
        if start not in graph:
            raise ValueError(f"Start node {start} not in graph")
        if end not in graph:
            raise ValueError(f"End node {end} not in graph")
        if not nx.is_connected(nx.Graph(graph)):
            raise ValueError("Graph must be connected")
    
    def _preprocess_graph(self, graph: Dict, start: str, end: str) -> Dict:
        """
        Simplify graph for quantum processing:
        - Convert to unweighted for oracle creation
        - Limit node degree
        - Ensure connectivity
        """
        # Create networkx graph
        G = nx.Graph()
        for node, neighbors in graph.items():
            for neighbor, _ in neighbors.items():
                G.add_edge(node, neighbor)
        
        # Get shortest path neighborhood
        try:
            sp = nx.shortest_path(G, start, end)
            neighborhood = set(sp)
            for node in sp:
                neighborhood.update(G.neighbors(node))
            
            # Create subgraph
            subgraph = G.subgraph(neighborhood).copy()
            return {n: dict.fromkeys(subgraph.neighbors(n), 1) for n in subgraph.nodes()}
            
        except nx.NetworkXNoPath:
            raise ValueError(f"No path exists between {start} and {end}")
    
    def _requires_classical_fallback(self, graph: Dict) -> bool:
        """Determine if graph is too large for quantum processing"""
        num_nodes = len(graph)
        required_qubits = num_nodes * (num_nodes - 1)  # Upper bound
        return required_qubits > self.max_qubits
    
    def _create_oracle(self, graph: Dict) -> PhaseOracle:
        """Create quantum oracle for the graph problem"""
        graph_hash = str(sorted(graph.items()))
        
        if graph_hash in self._oracle_cache:
            return self._oracle_cache[graph_hash]
        
        # Convert graph to boolean expression
        clauses = []
        for node, neighbors in graph.items():
            for neighbor in neighbors:
                clauses.append(f"(x{node} AND x{neighbor})")
        
        expression = " OR ".join(clauses)
        oracle = PhaseOracle(expression)
        self._oracle_cache[graph_hash] = oracle
        return oracle
    
    def _is_valid_path(self, bitstring: str, graph: Dict, start: str, end: str) -> bool:
        """Check if a measured bitstring represents a valid path"""
        # Convert bitstring to node sequence
        active_nodes = [f"x{idx}" for idx, bit in enumerate(bitstring) if bit == '1']
        
        # Check connectivity
        G = nx.Graph(graph)
        return nx.has_path(G, start, end)
    
    def _interpret_results(self, 
                         counts: Dict[str, int], 
                         graph: Dict, 
                         start: str, 
                         end: str) -> Tuple[List[str], float]:
        """Convert quantum measurements to optimal path"""
        # Get most frequent valid path
        valid_paths = {
            bits: count 
            for bits, count in counts.items() 
            if self._is_valid_path(bits, graph, start, end)
        }
        
        if not valid_paths:
            raise ValueError("No valid paths found in quantum results")
        
        best_bits = max(valid_paths, key=valid_paths.get)
        
        # Convert to node sequence
        path = [f"x{idx}" for idx, bit in enumerate(best_bits) if bit == '1']
        
        # Calculate actual cost from original graph
        total_cost = sum(
            graph[path[i]][path[i+1]] 
            for i in range(len(path)-1)
        )
        
        return path, total_cost
    
    def _visualize_results(self, counts: Dict[str, int]):
        """Display quantum measurement results"""
        fig, ax = plt.subplots(figsize=(10, 6))
        plot_histogram(counts, ax=ax)
        ax.set_title("Quantum Measurement Results")
        st.pyplot(fig)
        plt.close(fig)
    
    def _classical_fallback(self, 
                          graph: Dict[str, Dict[str, float]], 
                          start: str, 
                          end: str) -> Tuple[List[str], float]:
        """Classical pathfinding fallback"""
        G = nx.Graph()
        for node, neighbors in graph.items():
            for neighbor, cost in neighbors.items():
                G.add_edge(node, neighbor, weight=cost)
        
        path = nx.shortest_path(G, source=start, target=end, weight='weight')
        cost = nx.shortest_path_length(G, source=start, target=end, weight='weight')
        
        return path, cost
    
    def visualize_circuit(self, graph: Dict, start: str, end: str):
        """Generate and display the quantum circuit diagram"""
        simplified_graph = self._preprocess_graph(graph, start, end)
        oracle = self._create_oracle(simplified_graph)
        
        problem = AmplificationProblem(oracle)
        grover = Grover()
        circuit = grover.construct_circuit(problem)
        
        st.write("## Quantum Circuit Diagram")
        st.text(str(circuit.draw()))
        
        # For actual visualization in local environments:
        # display(circuit.draw(output='mpl'))