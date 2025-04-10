import heapq

def heuristic(node, goal):
    """Heuristic function: Estimates distance from node to goal."""
    return abs(ord(node) - ord(goal))  # Simple heuristic based on letter difference

def a_star(graph, start, end):
    """Finds the shortest path using A* Algorithm."""
    priority_queue = [(0, start)]  # (f_cost, node)
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    previous_nodes = {}
    
    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)
        
        if current_node == end:
            path = []
            while current_node in previous_nodes:
                path.insert(0, current_node)
                current_node = previous_nodes[current_node]
            path.insert(0, start)
            return path, distances[end]
        
        for neighbor, weight in graph[current_node].items():
            g_cost = distances[current_node] + weight
            h_cost = heuristic(neighbor, end)
            f_cost = g_cost + h_cost
            
            if g_cost < distances[neighbor]:
                distances[neighbor] = g_cost
                previous_nodes[neighbor] = current_node
                heapq.heappush(priority_queue, (f_cost, neighbor))
    
    return None, float('inf')  # No path found

# Example Usage
if __name__ == "__main__":
    graph = {
        'A': {'B': 4, 'C': 1},
        'B': {'A': 4, 'C': 2, 'D': 5},
        'C': {'A': 1, 'B': 2, 'D': 8},
        'D': {'B': 5, 'C': 8}
    }
    start, end = 'A', 'D'
    path, cost = a_star(graph, start, end)
    print(f"Optimal Path: {path}, Cost: {cost}")
  
