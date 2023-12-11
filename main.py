import csv
import random
import heapq
import time
from collections import deque



class Graph:
    def __init__(self):
        self.nodes = {}
        self.edges = {}

    def add_node(self, node, x, y):
        self.nodes[node] = (x, y)

    def add_edge(self, u, v, capacity):
        if u not in self.edges:
            self.edges[u] = []
        self.edges[u].append((v, capacity))


    def display_graph(self):
        for i in self.edges:
            print(str(i) + ": " + str(self.edges[i]))

    def to_csv(self, n, r, upper_cap):
        filename = "generated_graph_n{}_r{}_cap{}.csv".format(n, r, upper_cap)
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Node", "X", "Y"])
            for node, (x, y) in self.nodes.items():
                writer.writerow([node, x, y])

            writer.writerow([])
            writer.writerow(["Edge", "Source", "Target", "Capacity"])
            for source, edges in self.edges.items():
                for target, capacity in edges:
                    writer.writerow(["", source, target, capacity])


    @classmethod
    def from_csv(cls, n, r, upper_cap):
        filename = "generated_graph_n{}_r{}_cap{}.csv".format(n, r, upper_cap)
        graph = cls()
        with open(filename, mode='r') as file:
            reader = csv.reader(file)
            node_section = True
            edge_section = False
            for row in reader:
                if not row:
                    node_section = not node_section
                    edge_section = not edge_section
                    continue

                if node_section and row[0] != 'Node':
                    node, x, y = map(float, row)
                    graph.add_node(node, x, y)
                elif edge_section and row[0] == '':
                    _, source, target, capacity = row
                    graph.add_edge(int(source), int(target), int(capacity))

        return graph

def euclidean_distance(node1, node2):
    x1, y1 = node1
    x2, y2 = node2
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

def generate_graph(n, r, upper_cap):
    graph = Graph()

    for i in range(n):
        x = random.uniform(0, 1)
        y = random.uniform(0, 1)
        graph.add_node(i, x, y)

    for u in range(n):
        for v in range(u + 1, n):
            if euclidean_distance(graph.nodes[u], graph.nodes[v]) <= r:
                rand = random.uniform(0, 1)
                if rand < 0.5:
                    if (u, v) not in graph.edges and (v, u) not in graph.edges:
                        graph.add_edge(u, v, random.randint(1, upper_cap))
                else:
                    if (u, v) not in graph.edges and (v, u) not in graph.edges:
                        graph.add_edge(v, u, random.randint(1, upper_cap))


    source = random.choice(list(graph.edges.keys()))

    visited = set()
    queue = deque([(source, [source])])

    while queue:
        node, path = queue.popleft()


        if node not in visited:
            visited.add(node)
            neighbors = [v for v, _ in graph.edges.get(node, [])]
            for neighbor in neighbors:
                if neighbor not in path:
                    new_path = path + [neighbor]
                    queue.append((neighbor, new_path))

    sink = path[-1]
    graph.to_csv(n, r, upper_cap)

    return graph, source, sink

def FordFulkerson(graph, source, sink, augmenting_algorithm):
    paths = 0
    total_length = 0
    total_proportional_length = 0

    while True:
        augmenting_path = augmenting_algorithm(graph, source, sink)

        if not augmenting_path:
            break

        paths += 1
        total_length += len(augmenting_path)
        total_proportional_length += len(augmenting_path) / len(augmenting_algorithm(graph, source, sink))
        for i in range(len(augmenting_path) - 1):
            u, v = augmenting_path[i], augmenting_path[i + 1]
            edge_found = False
            for j, (neighbor, edge_capacity) in enumerate(graph.edges[u]):
                if neighbor == v and edge_capacity > 0:
                    graph.edges[u][j] = (v, edge_capacity - 1)
                    edge_found = True
                    break
            if not edge_found:
                print("Error: No valid edge found for path {}".format(augmenting_path))
                break

    return paths, total_length, total_proportional_length


def Djikstra(graph, source, sink):
    distance = {node: float('inf') for node in graph.nodes}
    distance[source] = 0
    previous = {node: None for node in graph.nodes}
    min_heap = [(0, source)]

    while min_heap:
        current_dist, current_node = heapq.heappop(min_heap)

        if current_node == sink:
            path = []
            while current_node is not None:
                path.insert(0, current_node)
                current_node = previous[current_node]
            return path

        if current_dist > distance[current_node]:
            continue

        if current_node in graph.edges:
            for neighbor, edge_capacity in graph.edges[current_node]:
                if edge_capacity > 0:
                    total_distance = current_dist + edge_capacity
                    if total_distance < distance[neighbor]:
                        distance[neighbor] = total_distance
                        previous[neighbor] = current_node
                        heapq.heappush(min_heap, (total_distance, neighbor))

    return None


def SAP(graph, source, sink):
    distance = {node: float('inf') for node in graph.nodes}
    distance[source] = 0
    previous = {node: None for node in graph.nodes}
    min_heap = [(0, source)]

    while min_heap:
        current_dist, current_node = heapq.heappop(min_heap)

        if current_node == sink:
            path = []
            while current_node is not None:
                path.insert(0, current_node)
                current_node = previous[current_node]
            return path

        if current_dist > distance[current_node]:
            continue

        if current_node in graph.edges:
            for neighbor, edge_capacity in graph.edges[current_node]:
                if edge_capacity > 0:
                    total_distance = current_dist + 1
                    if total_distance < distance[neighbor]:
                        distance[neighbor] = total_distance
                        previous[neighbor] = current_node
                        heapq.heappush(min_heap, (total_distance, neighbor))

    return None



def DFS(graph, source, sink):
    distances = {node: float('inf') for node in graph.nodes}
    distances[source] = 0

    visited = set()
    path = []

    priority_queue = [(0, source)]
    counter = 999999

    def decrease_key(node):
        nonlocal counter
        if distances[node] == float('inf'):
            counter -= 1
            distances[node] = counter

    def dfs(node):
        nonlocal path
        visited.add(node)
        decrease_key(node)

        if node == sink:
            return True

        if node in graph.edges:
            for neighbor, edge_capacity in graph.edges[node]:
                if edge_capacity > 0 and neighbor not in visited:
                    decrease_key(neighbor)
                    path.append((node, neighbor))
                    if dfs(neighbor):
                        return True
                    path.pop()

        return False

    if dfs(source):
        result_path = [source]
        for edge in path:
            result_path.append(edge[1])
        return result_path
    else:
        return None


def MaxCap(graph, source, sink):
    distance = {node: -float('inf') for node in graph.nodes}
    distance[source] = float('inf')
    previous = {node: None for node in graph.nodes}
    min_heap = [(float('inf'), source)]

    while min_heap:
        current_dist, current_node = heapq.heappop(min_heap)

        if current_node == sink:
            path = []
            while current_node is not None:
                path.insert(0, current_node)
                current_node = previous[current_node]
            return path  # Return the augmenting path and critical edge capacity

        if current_dist > distance[current_node]:
            continue

        if current_node in graph.edges:
            for neighbor, edge_capacity in graph.edges[current_node]:
                if edge_capacity > 0:
                    min_capacity = min(edge_capacity, distance[current_node])
                    if min_capacity > distance[neighbor]:
                        distance[neighbor] = min_capacity
                        previous[neighbor] = current_node
                        heapq.heappush(min_heap, (-min_capacity, neighbor))

    return None# Return None if no augmenting path is found

def Random(graph, source, sink):
    distance = {node: float('inf') for node in graph.nodes}
    distance[source] = 0
    previous = {node: None for node in graph.nodes}
    min_heap = [(0, random.random(), source)]

    while min_heap:
        current_dist, _, current_node = heapq.heappop(min_heap)

        if current_node == sink:
            path = []
            while current_node is not None:
                path.insert(0, current_node)
                current_node = previous[current_node]
            return path

        if current_dist > distance[current_node]:
            continue

        if current_node in graph.edges:
            for neighbor, edge_capacity in graph.edges[current_node]:
                if edge_capacity > 0:
                    total_distance = current_dist + edge_capacity
                    if total_distance < distance[neighbor]:
                        distance[neighbor] = total_distance
                        previous[neighbor] = current_node
                        heapq.heappush(min_heap, (total_distance, random.random(), neighbor))

    return None




def run_simulations(n, r, upper_cap):
    graph, source, sink = generate_graph(n, r, upper_cap)
    augmenting_algorithms = [SAP, DFS, MaxCap, Random]
    total_edges = sum(len(edges) for edges in graph.edges.values())

    for algorithm in augmenting_algorithms:
        saved_graph = Graph.from_csv(n, r, upper_cap)  # Load graph from the CSV file
        start_time = time.time()
        paths, total_length, total_proportional_length = FordFulkerson(saved_graph, source, sink, algorithm)
        elapsed_time = time.time() - start_time

        mean_length = total_length / paths if paths > 0 else 0
        mean_proportional_length = total_proportional_length / paths if paths > 0 else 0
        print("Algorithm name: {}".format(algorithm.__name__))
        print("Paths: {}".format(paths))
        print("Mean Length: {}".format(mean_length))
        print("Mean Proprotional Length: {}".format(mean_proportional_length))
        print("Total Edges: {}".format(total_edges))
        print()



if __name__ == "__main__":
    simulations_params = [
        (100, 0.2, 2),
        (200, 0.2, 2),
        (100, 0.3, 2),
        (200, 0.3, 2),
        (100, 0.2, 50),
        (200, 0.2, 50),
        (100, 0.3, 50),
        (200, 0.3, 50),
        (20, 0.1, 5),
        (300, 0.5, 20)
    ]

    for n, r, upper_cap in simulations_params:
        print("\nSimulation for n={}, r={}, upperCap={}".format(n, r, upper_cap))
        run_simulations(n, r, upper_cap)
