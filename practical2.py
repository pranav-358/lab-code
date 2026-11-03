                  #BFS AND DFS
from collections import deque
import time

# Graph representation using an adjacency list
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D'],
    'C': ['A', 'E'],
    'D': ['B', 'E'],
    'E': ['C', 'D']
}


# Breadth First Search
def bfs(graph, start, goal):
    queue = deque([start])
    visited = {start}
    parent = {start: None}
    nodes_explored = 0

    while queue:
        node = queue.popleft()
        nodes_explored += 1

        if node == goal:
            break

        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                parent[neighbor] = node
                queue.append(neighbor)

    if goal not in parent:
        return None, nodes_explored

    path = []
    node = goal

    while node is not None:
        path.append(node)
        node = parent[node]

    path.reverse()
    return path, nodes_explored


# Depth First Search
def dfs(graph, start, goal):
    visited = set()
    path = []
    nodes_explored = 0

    def search(node):
        nonlocal nodes_explored

        visited.add(node)
        path.append(node)
        nodes_explored += 1

        if node == goal:
            return True

        for neighbor in graph[node]:
            if neighbor not in visited:
                if search(neighbor):
                    return True

        path.pop()
        return False

    if search(start):
        return path.copy(), nodes_explored

    return None, nodes_explored


# Input
start = 'A'
goal = 'E'

# Run BFS and measure execution time
start_time = time.perf_counter()
bfs_path, bfs_nodes = bfs(graph, start, goal)
bfs_time = time.perf_counter() - start_time

# Run DFS and measure execution time
start_time = time.perf_counter()
dfs_path, dfs_nodes = dfs(graph, start, goal)
dfs_time = time.perf_counter() - start_time

# Display results
print("BREADTH FIRST SEARCH (BFS)")
print("Path:", bfs_path)
print("Path Length:", len(bfs_path) - 1 if bfs_path else 0)
print("Nodes Explored:", bfs_nodes)
print("Execution Time:", bfs_time, "seconds")

print("\nDEPTH FIRST SEARCH (DFS)")
print("Path:", dfs_path)
print("Path Length:", len(dfs_path) - 1 if dfs_path else 0)
print("Nodes Explored:", dfs_nodes)
print("Execution Time:", dfs_time, "seconds")
