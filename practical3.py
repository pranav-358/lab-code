
import heapq
import time

# Weighted graph
graph = {
    'A': [('B', 1), ('C', 4)],
    'B': [('D', 5), ('E', 2)],
    'C': [('E', 1), ('F', 5)],
    'D': [('G', 3)],
    'E': [('G', 3)],
    'F': [('G', 2)],
    'G': []
}

# Heuristic values to goal G
heuristic = {
    'A': 5,
    'B': 4,
    'C': 4,
    'D': 3,
    'E': 2,
    'F': 1,
    'G': 0
}


# Greedy Best-First Search
def greedy_best_first_search(start, goal):
    priority_queue = [(heuristic[start], start, [start])]
    visited = set()
    explored = 0

    while priority_queue:
        h, node, path = heapq.heappop(priority_queue)

        if node in visited:
            continue

        visited.add(node)
        explored += 1

        if node == goal:
            return path, explored

        for neighbor, cost in graph[node]:
            if neighbor not in visited:
                heapq.heappush(
                    priority_queue,
                    (heuristic[neighbor], neighbor, path + [neighbor])
                )

    return None, explored


# A* Search
def a_star_search(start, goal):
    priority_queue = [(heuristic[start], 0, start, [start])]
    best_cost = {start: 0}
    explored = 0

    while priority_queue:
        f, g, node, path = heapq.heappop(priority_queue)

        if g > best_cost.get(node, float('inf')):
            continue

        explored += 1

        if node == goal:
            return path, g, explored

        for neighbor, cost in graph[node]:
            new_g = g + cost

            if new_g < best_cost.get(neighbor, float('inf')):
                best_cost[neighbor] = new_g
                new_f = new_g + heuristic[neighbor]

                heapq.heappush(
                    priority_queue,
                    (new_f, new_g, neighbor, path + [neighbor])
                )

    return None, float('inf'), explored


# Input
start = input("Enter Start City (A-G): ").strip().upper()
goal = input("Enter Destination City (A-G): ").strip().upper()

if start not in graph or goal not in graph:
    print("Invalid city name!")

else:
    # Greedy Best-First Search
    t1 = time.perf_counter()
    greedy_path, greedy_nodes = greedy_best_first_search(start, goal)
    t2 = time.perf_counter()

    print("\n--- Greedy Best-First Search ---")

    if greedy_path:
        print("Path:", " -> ".join(greedy_path))
        print("Path Cost:", sum(
            next(cost for neighbor, cost in graph[node]
                 if neighbor == greedy_path[i + 1])
            for i, node in enumerate(greedy_path[:-1])
        ))
    else:
        print("No path found.")

    print("Nodes Explored:", greedy_nodes)
    print("Execution Time: {:.6f} ms".format((t2 - t1) * 1000))

    # A* Search
    t3 = time.perf_counter()
    astar_path, astar_cost, astar_nodes = a_star_search(start, goal)
    t4 = time.perf_counter()

    print("\n--- A* Search ---")

    if astar_path:
        print("Path:", " -> ".join(astar_path))
        print("Path Cost:", astar_cost)
    else:
        print("No path found.")

    print("Nodes Explored:", astar_nodes)
    print("Execution Time: {:.6f} ms".format((t4 - t3) * 1000))	
