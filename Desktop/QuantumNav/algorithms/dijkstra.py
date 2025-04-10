import heapq

def dijkstra(graph, start, end):
    """Finds the shortest path between start and end nodes using Dijkstra's Algorithm."""
    priority_queue = [(0, start)]  # (distance, node)
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
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                previous_nodes[neighbor] = current_node
                heapq.heappush(priority_queue, (distance, neighbor))

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
    path, cost = dijkstra(graph, start, end)
    print(f"Dijkstra Shortest Path: {path}, Cost: {cost}")

