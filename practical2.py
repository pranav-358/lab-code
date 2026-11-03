"""
PR2: UNINFORMED SEARCH ALGORITHMS (CO1, CO2)
Title: Route Finding using Breadth First Search (BFS) and Depth First Search (DFS)

Description:
This program finds a route between two locations in an undirected graph.
It implements BFS and DFS and compares:
1. Path length
2. Number of nodes explored
3. Execution time

Run:
    python PR2_BFS_DFS.py

No external libraries are required.
"""

from collections import deque
from time import perf_counter


# ============================================================
# 1. GRAPH REPRESENTATION
# ============================================================

# Each location is a node and each connection is an edge.
# The graph is undirected.
graph = {
    "A": ["B", "C"],
    "B": ["A", "D", "E"],
    "C": ["A", "F"],
    "D": ["B", "G"],
    "E": ["B", "G"],
    "F": ["C", "G"],
    "G": ["D", "E", "F"]
}


# ============================================================
# 2. BREADTH FIRST SEARCH (BFS)
# ============================================================

def bfs(graph, start, goal):
    """
    Find the shortest path using Breadth First Search.

    BFS explores nodes level by level using a queue.
    It returns the path, number of explored nodes and execution time.
    """
    queue = deque([start])
    visited = {start}
    parent = {start: None}
    nodes_explored = 0

    while queue:
        current = queue.popleft()
        nodes_explored += 1

        if current == goal:
            break

        for neighbour in graph[current]:
            if neighbour not in visited:
                visited.add(neighbour)
                parent[neighbour] = current
                queue.append(neighbour)

    # If the goal was not reached, no path exists.
    if goal not in parent:
        return None, nodes_explored

    # Reconstruct the path from goal to start.
    path = []
    current = goal

    while current is not None:
        path.append(current)
        current = parent[current]

    path.reverse()
    return path, nodes_explored


# ============================================================
# 3. DEPTH FIRST SEARCH (DFS)
# ============================================================

def dfs(graph, start, goal):
    """
    Find a path using Depth First Search.

    DFS explores one branch as deeply as possible before backtracking.
    It returns the path and number of explored nodes.
    """
    visited = set()
    path = []
    nodes_explored = 0

    def search(current):
        nonlocal nodes_explored

        visited.add(current)
        path.append(current)
        nodes_explored += 1

        if current == goal:
            return True

        for neighbour in graph[current]:
            if neighbour not in visited:
                if search(neighbour):
                    return True

        # Backtrack if this branch does not reach the goal.
        path.pop()
        return False

    if search(start):
        return path.copy(), nodes_explored

    return None, nodes_explored


# ============================================================
# 4. DISPLAY GRAPH
# ============================================================

def display_graph():
    """Display all locations and their connections."""
    print("\nGRAPH (LOCATION CONNECTIONS)")
    print("-" * 40)

    for location, neighbours in graph.items():
        print(f"{location} -> {', '.join(neighbours)}")

    print("-" * 40)


# ============================================================
# 5. GET VALID START AND GOAL LOCATIONS
# ============================================================

def get_location(prompt):
    """Accept a location only if it exists in the graph."""
    while True:
        location = input(prompt).strip().upper()

        if location in graph:
            return location

        print("Invalid location. Available locations:", ", ".join(graph))


# ============================================================
# 6. RUN AND COMPARE BFS AND DFS
# ============================================================

def compare_search_algorithms(start, goal):
    """Run both algorithms and display their performance."""

    # Run BFS and measure execution time.
    bfs_start_time = perf_counter()
    bfs_path, bfs_nodes = bfs(graph, start, goal)
    bfs_time = perf_counter() - bfs_start_time

    # Run DFS and measure execution time.
    dfs_start_time = perf_counter()
    dfs_path, dfs_nodes = dfs(graph, start, goal)
    dfs_time = perf_counter() - dfs_start_time

    print("\n" + "=" * 55)
    print("          BREADTH FIRST SEARCH (BFS)")
    print("=" * 55)

    if bfs_path:
        print("Path             :", " -> ".join(bfs_path))
        print("Path Length      :", len(bfs_path) - 1, "edges")
    else:
        print("Path             : No path found")
        print("Path Length      : Not available")

    print("Nodes Explored   :", bfs_nodes)
    print(f"Execution Time   : {bfs_time:.8f} seconds")

    print("\n" + "=" * 55)
    print("          DEPTH FIRST SEARCH (DFS)")
    print("=" * 55)

    if dfs_path:
        print("Path             :", " -> ".join(dfs_path))
        print("Path Length      :", len(dfs_path) - 1, "edges")
    else:
        print("Path             : No path found")
        print("Path Length      : Not available")

    print("Nodes Explored   :", dfs_nodes)
    print(f"Execution Time   : {dfs_time:.8f} seconds")

    # Comparison table.
    print("\n" + "=" * 65)
    print("                 PERFORMANCE COMPARISON")
    print("=" * 65)
    print(f"{'Metric':<22}{'BFS':<21}{'DFS':<21}")
    print("-" * 65)

    bfs_length = len(bfs_path) - 1 if bfs_path else "Not found"
    dfs_length = len(dfs_path) - 1 if dfs_path else "Not found"

    print(f"{'Path Length':<22}{str(bfs_length):<21}{str(dfs_length):<21}")
    print(f"{'Nodes Explored':<22}{str(bfs_nodes):<21}{str(dfs_nodes):<21}")
    print(f"{'Time (seconds)':<22}{f'{bfs_time:.8f}':<21}{f'{dfs_time:.8f}':<21}")
    print("=" * 65)

    print("\nCONCLUSION")
    print("1. BFS explores nodes level by level and finds a shortest path")
    print("   when all edges have equal cost.")
    print("2. DFS explores one branch deeply before backtracking.")
    print("3. DFS does not always find the shortest path.")
    print("4. Execution time depends on the graph and search order.")


# ============================================================
# 7. MAIN PROGRAM
# ============================================================

def main():
    """Main function to execute the route-finding program."""

    print("\n" + "=" * 55)
    print("       ROUTE FINDING USING BFS AND DFS")
    print("=" * 55)

    display_graph()

    print("\nEnter the starting and destination locations.")
    start = get_location("Starting location: ")
    goal = get_location("Destination location: ")

    compare_search_algorithms(start, goal)

    print("\nProgram completed successfully.")


# ============================================================
# 8. PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
