"""
PR3: INFORMED SEARCH ALGORITHMS (CO1, CO2)
Title: Route Finding using Greedy Best-First Search and A* Search

This program:
1. Represents a weighted road network as a graph.
2. Uses Greedy Best-First Search and A* Search to find a route.
3. Displays the route, total path cost, explored nodes and execution time.
4. Compares the optimality, completeness and efficiency of both algorithms.

Run:
    python PR3_Greedy_AStar.py

No external libraries are required.
"""

import heapq
from itertools import count
from time import perf_counter


# ============================================================
# 1. WEIGHTED GRAPH
# ============================================================

# Each edge is represented as (neighbour, travel_cost).
# The graph is directed: each listed connection is one-way.
graph = {
    "A": [("B", 1), ("C", 4)],
    "B": [("D", 5), ("E", 2)],
    "C": [("E", 1), ("F", 5)],
    "D": [("G", 3)],
    "E": [("G", 3)],
    "F": [("G", 2)],
    "G": []
}

# Estimated remaining cost from each city to destination G.
# These values are admissible and consistent for this graph.
heuristic = {
    "A": 5,
    "B": 4,
    "C": 3,
    "D": 3,
    "E": 2,
    "F": 1,
    "G": 0
}


# ============================================================
# 2. HELPER FUNCTION
# ============================================================

def calculate_path_cost(path):
    """Calculate the total travel cost of a path."""
    if not path or len(path) == 1:
        return 0

    total_cost = 0

    for index in range(len(path) - 1):
        current_city = path[index]
        next_city = path[index + 1]

        for neighbour, cost in graph[current_city]:
            if neighbour == next_city:
                total_cost += cost
                break

    return total_cost


def display_graph():
    """Display all cities and their outgoing connections."""
    print("\nWEIGHTED ROAD NETWORK")
    print("-" * 45)

    for city, connections in graph.items():
        routes = ", ".join(
            f"{neighbour} (cost {cost})"
            for neighbour, cost in connections
        )
        print(f"{city} -> {routes if routes else 'No outgoing route'}")

    print("-" * 45)


# ============================================================
# 3. GREEDY BEST-FIRST SEARCH
# ============================================================

def greedy_best_first_search(start, goal):
    """
    Greedy Best-First Search selects the next city using h(n).

    It prioritizes the city that appears closest to the goal.
    It does not consider the path cost already travelled.
    """
    priority_queue = []
    tie_breaker = count()

    heapq.heappush(
        priority_queue,
        (heuristic[start], next(tie_breaker), start, [start])
    )

    visited = set()
    nodes_explored = 0

    while priority_queue:
        h_value, _, current_city, path = heapq.heappop(priority_queue)

        if current_city in visited:
            continue

        visited.add(current_city)
        nodes_explored += 1

        if current_city == goal:
            return path, nodes_explored

        for neighbour, edge_cost in graph[current_city]:
            if neighbour not in visited:
                heapq.heappush(
                    priority_queue,
                    (
                        heuristic[neighbour],
                        next(tie_breaker),
                        neighbour,
                        path + [neighbour]
                    )
                )

    return None, nodes_explored


# ============================================================
# 4. A* SEARCH
# ============================================================

def a_star_search(start, goal):
    """
    A* Search selects the next city using f(n) = g(n) + h(n).

    g(n) = actual cost from the start city to the current city.
    h(n) = estimated cost from the current city to the goal.

    The best known cost is stored for each city. Better routes can
    update a city's cost, and outdated queue entries are ignored.
    """
    priority_queue = []
    tie_breaker = count()

    heapq.heappush(
        priority_queue,
        (heuristic[start], next(tie_breaker), 0, start, [start])
    )

    best_cost = {start: 0}
    nodes_explored = 0

    while priority_queue:
        f_value, _, current_cost, current_city, path = heapq.heappop(
            priority_queue
        )

        # Ignore an outdated queue entry.
        if current_cost > best_cost.get(current_city, float("inf")):
            continue

        nodes_explored += 1

        if current_city == goal:
            return path, current_cost, nodes_explored

        for neighbour, edge_cost in graph[current_city]:
            new_cost = current_cost + edge_cost

            if new_cost < best_cost.get(neighbour, float("inf")):
                best_cost[neighbour] = new_cost
                new_f_value = new_cost + heuristic[neighbour]

                heapq.heappush(
                    priority_queue,
                    (
                        new_f_value,
                        next(tie_breaker),
                        new_cost,
                        neighbour,
                        path + [neighbour]
                    )
                )

    return None, float("inf"), nodes_explored


# ============================================================
# 5. INPUT VALIDATION
# ============================================================

def get_city(prompt):
    """Read a city name and check whether it exists in the graph."""
    while True:
        city = input(prompt).strip().upper()

        if city in graph:
            return city

        print("Invalid city. Available cities:", ", ".join(graph.keys()))


# ============================================================
# 6. DISPLAY RESULTS AND COMPARISON
# ============================================================

def display_result(algorithm_name, path, path_cost, nodes_explored, elapsed_time):
    """Display the result of one search algorithm."""
    print("\n" + "=" * 55)
    print(f"          {algorithm_name}")
    print("=" * 55)

    if path is None:
        print("Path             : No route found")
        print("Path Cost        : Not available")
    else:
        print("Path             :", " -> ".join(path))
        print("Path Cost        :", path_cost)

    print("Nodes Explored   :", nodes_explored)
    print(f"Execution Time   : {elapsed_time * 1000:.6f} milliseconds")


def display_comparison(greedy_path, greedy_cost, greedy_nodes, greedy_time,
                       astar_path, astar_cost, astar_nodes, astar_time):
    """Compare the performance of Greedy Best-First Search and A*."""
    print("\n" + "=" * 72)
    print("                    PERFORMANCE COMPARISON")
    print("=" * 72)
    print(f"{'Metric':<22}{'Greedy Best-First':<25}{'A* Search':<25}")
    print("-" * 72)

    greedy_length = len(greedy_path) - 1 if greedy_path else "No path"
    astar_length = len(astar_path) - 1 if astar_path else "No path"
    greedy_cost_text = greedy_cost if greedy_path else "No path"
    astar_cost_text = astar_cost if astar_path else "No path"

    print(f"{'Path Length (edges)':<22}{str(greedy_length):<25}{str(astar_length):<25}")
    print(f"{'Total Path Cost':<22}{str(greedy_cost_text):<25}{str(astar_cost_text):<25}")
    print(f"{'Nodes Explored':<22}{str(greedy_nodes):<25}{str(astar_nodes):<25}")
    print(f"{'Time (ms)':<22}{f'{greedy_time * 1000:.6f}':<25}{f'{astar_time * 1000:.6f}':<25}")
    print("=" * 72)


def display_algorithm_analysis():
    """Explain optimality, completeness and efficiency."""
    print("\n" + "=" * 65)
    print("              ALGORITHM ANALYSIS")
    print("=" * 65)

    print("\nGREEDY BEST-FIRST SEARCH")
    print("Optimality  : Not guaranteed to find the lowest-cost path.")
    print("Completeness: Complete on a finite graph when repeated")
    print("              states are avoided as in this implementation.")
    print("Efficiency  : Can be fast when the heuristic guides it well,")
    print("              but may explore an unhelpful route.")

    print("\nA* SEARCH")
    print("Optimality  : Finds an optimal path when the heuristic is")
    print("              admissible; this program uses a consistent heuristic.")
    print("Completeness: Complete for this finite graph with positive costs.")
    print("Efficiency  : Uses both actual cost g(n) and estimated cost h(n).")
    print("              It may explore more nodes than Greedy, but aims")
    print("              to find the lowest-cost path.")

    print("\nFORMULA")
    print("Greedy Best-First Search: f(n) = h(n)")
    print("A* Search               : f(n) = g(n) + h(n)")
    print("=" * 65)


# ============================================================
# 7. MAIN PROGRAM
# ============================================================

def main():
    """Run both informed search algorithms and compare their results."""
    print("\n" + "=" * 60)
    print("       INFORMED SEARCH: GREEDY BEST-FIRST AND A*")
    print("=" * 60)

    display_graph()

    start = get_city("Enter start city (A-G): ")
    goal = get_city("Enter destination city (A-G): ")

    if start == goal:
        print("\nStart and destination are the same city.")

    # Run Greedy Best-First Search.
    start_time = perf_counter()
    greedy_path, greedy_nodes = greedy_best_first_search(start, goal)
    greedy_time = perf_counter() - start_time
    greedy_cost = calculate_path_cost(greedy_path)

    # Run A* Search.
    start_time = perf_counter()
    astar_path, astar_cost, astar_nodes = a_star_search(start, goal)
    astar_time = perf_counter() - start_time

    # Display individual results.
    display_result(
        "GREEDY BEST-FIRST SEARCH",
        greedy_path,
        greedy_cost,
        greedy_nodes,
        greedy_time
    )

    display_result(
        "A* SEARCH",
        astar_path,
        astar_cost if astar_path else None,
        astar_nodes,
        astar_time
    )

    # Display comparison and theoretical analysis.
    display_comparison(
        greedy_path, greedy_cost, greedy_nodes, greedy_time,
        astar_path, astar_cost if astar_path else None, astar_nodes, astar_time
    )

    display_algorithm_analysis()

    print("\nProgram completed successfully.")


# ============================================================
# 8. PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
