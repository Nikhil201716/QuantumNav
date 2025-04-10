from itertools import permutations

def brute_force_search(graph, start, end):
    """Finds the shortest path by checking all possible routes (Brute Force)."""
    nodes = list(graph.keys())
    all_paths = permutations(nodes)
    shortest_path = None
    min_cost = float('inf')

    for path in all_paths:
        if path[0] == start and path[-1] == end:
            cost = 0
            valid_path = True
            for i in range(len(path) - 1):
                if path[i+1] in graph[path[i]]:
                    cost += graph[path[i]][path[i+1]]
                else:
                    valid_path = False
                    break
            if valid_path and cost < min_cost:
                min_cost = cost
                shortest_path = path
    
    return shortest_path, min_cost if shortest_path else (None, float('inf'))

# Example Usage
if __name__ == "__main__":
    graph = {
        'A': {'B': 4, 'C': 1},
        'B': {'A': 4, 'C': 2, 'D': 5},
        'C': {'A': 1, 'B': 2, 'D': 8},
        'D': {'B': 5, 'C': 8}
    }
    start, end = 'A', 'D'
    path, cost = brute_force_search(graph, start, end)
    print(f"Brute Force Shortest Path: {path}, Cost: {cost}")
  
