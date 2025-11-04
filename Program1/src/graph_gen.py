import networkx as nx
import random
import math
from typing import Dict, Tuple, List

class GraphGenerator:
    def __init__(self, seed: int = None):
        if seed is not None:
            random.seed(seed)
        self.graph = nx.Graph()
        self.coordinates = {}
        
    def generate_grid(self, size: int, obstacle_density: float, 
                     connectivity: str = "4", weighted: bool = False,
                     seed: int = None) -> Tuple[nx.Graph, Dict]:
        """Generate grid world with obstacles"""
        if seed is not None:
            random.seed(seed)
            
        self.graph = nx.Graph()
        self.coordinates = {}
        
        # Create grid nodes (non-obstacles)
        nodes = []
        for x in range(size):
            for y in range(size):
                if random.random() > obstacle_density:
                    node_id = f"({x},{y})"
                    self.coordinates[node_id] = (x, y)
                    self.graph.add_node(node_id, pos=(x, y))
                    nodes.append((x, y, node_id))
        
        # Connect nodes based on connectivity
        for i, (x1, y1, node1) in enumerate(nodes):
            for j, (x2, y2, node2) in enumerate(nodes[i+1:], i+1):
                if connectivity == "4":  # Manhattan
                    if (abs(x1 - x2) == 1 and y1 == y2) or (abs(y1 - y2) == 1 and x1 == x2):
                        weight = random.uniform(1, 10) if weighted else 1.0
                        self.graph.add_edge(node1, node2, weight=weight)
                else:  # 8-connectivity
                    if abs(x1 - x2) <= 1 and abs(y1 - y2) <= 1 and (x1 != x2 or y1 != y2):
                        weight = random.uniform(1, 10) if weighted else math.sqrt((x1-x2)**2 + (y1-y2)**2)
                        self.graph.add_edge(node1, node2, weight=weight)
        
        return self.graph, self.coordinates
    
    def generate_random_graph(self, num_nodes: int, branching_factor: float,
                            weight_range: Tuple[float, float] = (1, 10),
                            seed: int = None) -> Tuple[nx.Graph, Dict]:
        """Generate random graph with Poisson branching"""
        if seed is not None:
            random.seed(seed)
            
        self.graph = nx.Graph()
        self.coordinates = {}
        
        # Generate random positions
        for i in range(num_nodes):
            node_id = f"Node_{i}"
            x, y = random.uniform(0, 100), random.uniform(0, 100)
            self.coordinates[node_id] = (x, y)
            self.graph.add_node(node_id, pos=(x, y))
        
        # Connect with Poisson-distributed branching
        nodes = list(self.graph.nodes())
        for node in nodes:
            # Poisson distribution for number of connections
            num_connections = random.poisson(branching_factor)
            num_connections = max(1, min(num_connections, num_nodes - 1))
            
            # Connect to random nodes
            possible_neighbors = [n for n in nodes if n != node and not self.graph.has_edge(node, n)]
            if len(possible_neighbors) > num_connections:
                neighbors = random.sample(possible_neighbors, num_connections)
            else:
                neighbors = possible_neighbors
                
            for neighbor in neighbors:
                weight = random.uniform(weight_range[0], weight_range[1])
                self.graph.add_edge(node, neighbor, weight=weight)
        
        return self.graph, self.coordinates
    
    def generate_maze(self, size: int, seed: int = None) -> Tuple[nx.Graph, Dict]:
        """Generate maze using randomized DFS"""
        if seed is not None:
            random.seed(seed)
            
        self.graph = nx.Graph()
        self.coordinates = {}
        
        # Initialize all grid positions
        grid = {}
        for x in range(size):
            for y in range(size):
                node_id = f"({x},{y})"
                self.coordinates[node_id] = (x, y)
                self.graph.add_node(node_id, pos=(x, y))
                grid[(x, y)] = node_id
        
        # Randomized DFS maze generation
        visited = set()
        stack = []
        
        start = (0, 0)
        visited.add(start)
        stack.append(start)
        
        while stack:
            current = stack[-1]
            x, y = current
            
            # Get unvisited neighbors
            neighbors = []
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx_, ny = x + dx, y + dy
                if 0 <= nx_ < size and 0 <= ny < size and (nx_, ny) not in visited:
                    neighbors.append((nx_, ny))
            
            if neighbors:
                next_cell = random.choice(neighbors)
                # Add edge between current and next
                self.graph.add_edge(grid[current], grid[next_cell], weight=1.0)
                visited.add(next_cell)
                stack.append(next_cell)
            else:
                stack.pop()
        
        return self.graph, self.coordinates