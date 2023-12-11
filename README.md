
# Ford-Fulkerson Simulation with Various Augmenting Algorithms
## Introduction
This Python code implements a simulation of the Ford-Fulkerson algorithm for maximum flow in a network. The simulation includes the use of different augmenting path algorithms such as Shortest Augmenting Path (SAP), Depth-First Search (DFS), Maximum Capacity (MaxCap), and a Random augmenting path algorithm.

## Code Structure
The code consists of the following components:

**Graph Class**: Defines a graph with nodes and edges, and provides methods for adding nodes, edges, and displaying the graph. It also includes methods for saving and loading graphs to and from CSV files.

**Euclidean Distance Function**: Calculates the Euclidean distance between two points in the graph.

**Graph Generation Function**: Generates a random graph with specified parameters, including the number of nodes (n), maximum distance between nodes (r), and upper capacity for edges.

**Ford-Fulkerson Function**: Implements the Ford-Fulkerson algorithm with various augmenting path algorithms. It also calculates statistics such as the number of paths, mean length, and mean proportional length.

**Augmenting Path Algorithms**:

**SAP (Shortest Augmenting Path)**: Augmenting path algorithm based on shortest paths with capacity equal to 1.     
**DFS (Depth-First Search)**: Augmenting path algorithm using depth-first search.     
**MaxCap (Maximum Capacity)**: Augmenting path algorithm maximizing edge capacity.     
**Random**: Augmenting path algorithm with a random element.      

**Simulation Function**: Runs simulations for various graph sizes and parameters, measuring the performance of the Ford-Fulkerson algorithm with different augmenting path algorithms.

**Main Section**: Contains simulation parameters and runs simulations for different graph sizes and characteristics.

Usage
To run the simulations, execute the script, and the results for each simulation scenario will be printed, including the algorithm name, number of paths, mean length, mean proportional length, and total edges.      

## Simulation Scenarios       
The code includes simulations for different graph sizes (n), maximum distances between nodes (r), and upper capacities for edges. These scenarios provide insights into the performance of the Ford-Fulkerson algorithm with different augmenting path algorithms under varying graph conditions.

Feel free to modify the simulation parameters in the "simulations_params" list to explore additional scenarios.

## Conclusion
This code serves as a flexible tool for simulating and analyzing the Ford-Fulkerson algorithm's performance with various augmenting path algorithms in different network scenarios. It provides valuable insights into the efficiency and characteristics of different augmenting path strategies in the context of maximum flow problems.

## Compilation and Run

Login into the server computation.encs.concordia.ca

Run the command git clone https://github.com/mehtavandit/ALGO-Project
Go to the respective directory on the server and run the command git pull
Run the command python3 main.py.

