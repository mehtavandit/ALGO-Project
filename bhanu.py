import random
import heapq
import time
import csv
from collections import deque
# import matplotlib.pyplot as plt

class Graph:
    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.source = None
        self.sink = None

    def set_source(self, source):
        self.source = source

    def set_sink(self, sink):
        self.sink = sink

    def get_source(self):
        return self.source

    def get_sink(self):
        return self.sink

    def add_node(self, node, x, y):
        self.nodes[node] = (x, y)

    def add_edge(self, u, v, capacity):
        if u not in self.edges:
            self.edges[u] = []
        self.edges[u].append((v, capacity))

    def to_csv(self, n, r, upper_cap):
        filename = f"generated_graph_n{n}_r{r}_cap{upper_cap}.csv"
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

            writer.writerow([])
            writer.writerow(["Source", self.source])
            writer.writerow(["Sink", self.sink])

    @classmethod
    def from_csv(cls, n, r, upper_cap):
        filename = f"generated_graph_n{n}_r{r}_cap{upper_cap}.csv"
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

                if row[0] == 'Source':
                    graph.source = int(row[1])
                elif row[0] == 'Sink':
                    graph.sink = int(row[1])
                elif row[0] == 'Edge':
                    edge_section = True
                elif node_section and row[0] != 'Node':
                    try:
                        node, x, y = map(float, row)
                        graph.add_node(node, x, y)
                    except ValueError:
                        pass
                elif edge_section and row[0] == '':
                    try:
                        _, source, target, capacity = row
                        graph.add_edge(int(source), int(target), int(capacity))
                    except ValueError:
                        pass
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

    edges_created = False

    while not edges_created:
        graph.edges = {}
        for u in range(n):
            for v in range(u + 1, n):
                if euclidean_distance(graph.nodes[u], graph.nodes[v]) <= r:
                    rand = random.uniform(0, 1)
                    if rand < 0.5:
                        graph.add_edge(u, v, random.randint(1, upper_cap))
                    else:
                        graph.add_edge(v, u, random.randint(1, upper_cap))

        has_outgoing_edges = any(graph.edges.get(node) for node in range(n))

        if has_outgoing_edges:
            edges_created = True
        else:
            continue

    source_options = list(graph.edges.keys())
    source = random.choice(source_options)
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
    graph.set_sink(sink)
    graph.set_source(source)
    graph.to_csv(n, r, upper_cap)



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
                print(f"Error: No valid edge found for path {augmenting_path}")
                break

    return paths, total_length, total_proportional_length

def dijkstra(graph, source, sink):
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

def dfs_like(graph, source, sink):
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
            return path

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

    return None

def Randon_Dij(graph, source, sink):
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

def visualize_graph_from_csv(n, r, upper_cap):
    filename = f"generated_graph_n{n}_r{r}_cap{upper_cap}.csv"
    graph = Graph()

    with open(filename, 'r') as file:
        reader = csv.reader(file)
        node_section = True
        edge_section = False

        for row in reader:
            if not row:
                node_section = not node_section
                edge_section = not edge_section
                continue

            if row[0] == 'Source':
                graph.source = int(row[1])
            elif row[0] == 'Sink':
                graph.sink = int(row[1])
            elif row[0] == 'Node':
                edge_section = False
            elif row[0] == 'Edge':
                node_section = False
                edge_section = True
            elif node_section:
                try:
                    node, x, y = map(float, row)
                    graph.add_node(node, x, y)
                except ValueError:
                    pass
            elif edge_section:
                try:
                    _, source, target, capacity = row
                    graph.add_edge(int(source), int(target), int(capacity))
                except ValueError:
                    pass

    for node, (x, y) in graph.nodes.items():
        plt.scatter(x, y, label=str(node), s=100, color='blue')

    for source, targets in graph.edges.items():
        x1, y1 = graph.nodes[source]
        for target, _ in targets:
            x2, y2 = graph.nodes[target]
            plt.plot([x1, x2], [y1, y2], 'k-', alpha=0.5)

    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Graph Visualization')
    plt.legend()

    # Save the plot as a PNG file
    plt.savefig(f"generated_graph_n{n}_r{r}_cap{upper_cap}.png")

    # Display the plot
    # plt.show()



def run_simulations(n, r, upper_cap):
    generate_graph(n, r, upper_cap)
    augmenting_algorithms = [SAP, dfs_like, MaxCap, Randon_Dij]

    # Print header for the table
    print("Algorithms\t Paths\t ML\t MPL\t Total Edges")

    for algorithm in augmenting_algorithms:
        saved_graph = Graph.from_csv(n, r, upper_cap)
        total_edges = sum(len(edges) for edges in saved_graph.edges.values())
        start_time = time.time()
        paths, total_length, total_proportional_length = FordFulkerson(saved_graph, saved_graph.source, saved_graph.sink, algorithm)
        elapsed_time = time.time() - start_time

        mean_length = total_length / paths if paths > 0 else 0
        mean_proportional_length = total_proportional_length / paths if paths > 0 else 0

        # Print values in tabulated format
        print(f"{algorithm.__name__}\t {paths}\t {mean_length}\t {mean_proportional_length}\t {total_edges}")



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
        (150, 0.2, 100),
        (300, 0.4, 50),
        (150, 0.3, 20),
        (50, 0.1, 2),
    ]

    for n, r, upper_cap in simulations_params:
        print("\n_______________________________________________________________\n")
        print(f"\nSimulation for n={n}, r={r}, upperCap={upper_cap}")
        run_simulations(n, r, upper_cap)
        # visualize_graph_from_csv(n, r, upper_cap)
