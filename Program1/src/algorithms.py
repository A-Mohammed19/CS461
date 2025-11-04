import time
import heapq
from collections import deque
from typing import List, Tuple, Dict, Any, Optional
import networkx as nx

class SearchAlgorithm:
    def __init__(self, graph: nx.Graph, coordinates: Dict[str, Tuple[float, float]]):
        self.graph = graph
        self.coordinates = coordinates
        self.visited = set()
        self.parent = {}
        self.path = []
        self.metrics = {
            'nodes_expanded': 0,
            'nodes_generated': 0,
            'max_frontier_size': 0,
            'max_explored_size': 0,
            'path_cost': 0,
            'solution_depth': 0,
            'execution_time': 0,
            'branching_factors': []
        }
        
    def search(self, start: str, goal: str) -> Tuple[List[str], Dict[str, Any]]:
        raise NotImplementedError
        
    def _calculate_path_cost(self, path: List[str]) -> float:
        if len(path) < 2:
            return 0.0
        total_cost = 0.0
        for i in range(len(path) - 1):
            if self.graph.has_edge(path[i], path[i+1]):
                total_cost += self.graph[path[i]][path[i+1]]['weight']
        return total_cost
    
    def _reconstruct_path(self, start: str, goal: str) -> List[str]:
        path = []
        current = goal
        while current != start:
            path.append(current)
            if current not in self.parent:
                return []
            current = self.parent[current]
        path.append(start)
        path.reverse()
        return path
    
    def _update_branching_factor(self, node: str):
        if node in self.graph:
            branching = len(list(self.graph.neighbors(node)))
            self.metrics['branching_factors'].append(branching)

class DFS(SearchAlgorithm):
    def search(self, start: str, goal: str) -> Tuple[List[str], Dict[str, Any]]:
        start_time = time.perf_counter()
        self.visited = set()
        self.parent = {}
        self.metrics = {k: 0 for k in self.metrics.keys()}
        self.metrics['branching_factors'] = []
        
        stack = [start]
        self.parent[start] = None
        self.visited.add(start)
        self.metrics['nodes_generated'] = 1
        
        while stack:
            self.metrics['max_frontier_size'] = max(self.metrics['max_frontier_size'], len(stack))
            current = stack.pop()
            self.metrics['nodes_expanded'] += 1
            self._update_branching_factor(current)
            
            if current == goal:
                self.path = self._reconstruct_path(start, goal)
                self.metrics['path_cost'] = self._calculate_path_cost(self.path)
                self.metrics['solution_depth'] = len(self.path) - 1 if self.path else 0
                self.metrics['max_explored_size'] = len(self.visited)
                self.metrics['execution_time'] = time.perf_counter() - start_time
                return self.path, self.metrics
            
            neighbors = list(self.graph.neighbors(current))
            for neighbor in neighbors:
                if neighbor not in self.visited:
                    self.visited.add(neighbor)
                    self.parent[neighbor] = current
                    stack.append(neighbor)
                    self.metrics['nodes_generated'] += 1
        
        self.metrics['execution_time'] = time.perf_counter() - start_time
        return [], self.metrics

class BFS(SearchAlgorithm):
    def search(self, start: str, goal: str) -> Tuple[List[str], Dict[str, Any]]:
        start_time = time.perf_counter()
        self.visited = set()
        self.parent = {}
        self.metrics = {k: 0 for k in self.metrics.keys()}
        self.metrics['branching_factors'] = []
        
        queue = deque([start])
        self.parent[start] = None
        self.visited.add(start)
        self.metrics['nodes_generated'] = 1
        
        while queue:
            self.metrics['max_frontier_size'] = max(self.metrics['max_frontier_size'], len(queue))
            current = queue.popleft()
            self.metrics['nodes_expanded'] += 1
            self._update_branching_factor(current)
            
            if current == goal:
                self.path = self._reconstruct_path(start, goal)
                self.metrics['path_cost'] = self._calculate_path_cost(self.path)
                self.metrics['solution_depth'] = len(self.path) - 1 if self.path else 0
                self.metrics['max_explored_size'] = len(self.visited)
                self.metrics['execution_time'] = time.perf_counter() - start_time
                return self.path, self.metrics
            
            for neighbor in self.graph.neighbors(current):
                if neighbor not in self.visited:
                    self.visited.add(neighbor)
                    self.parent[neighbor] = current
                    queue.append(neighbor)
                    self.metrics['nodes_generated'] += 1
        
        self.metrics['execution_time'] = time.perf_counter() - start_time
        return [], self.metrics

class IDDFS(SearchAlgorithm):
    def search(self, start: str, goal: str) -> Tuple[List[str], Dict[str, Any]]:
        start_time = time.perf_counter()
        self.metrics = {k: 0 for k in self.metrics.keys()}
        self.metrics['branching_factors'] = []
        
        depth = 0
        while depth <= 1000:  # Safety limit
            self.visited = set()
            self.parent = {}
            result = self._depth_limited_search(start, goal, depth)
            if result:
                self.path = self._reconstruct_path(start, goal)
                self.metrics['path_cost'] = self._calculate_path_cost(self.path)
                self.metrics['solution_depth'] = len(self.path) - 1 if self.path else 0
                self.metrics['max_explored_size'] = len(self.visited)
                self.metrics['execution_time'] = time.perf_counter() - start_time
                return self.path, self.metrics
            depth += 1
        
        self.metrics['execution_time'] = time.perf_counter() - start_time
        return [], self.metrics
    
    def _depth_limited_search(self, node: str, goal: str, depth: int) -> bool:
        if depth == 0:
            self.metrics['nodes_expanded'] += 1
            self._update_branching_factor(node)
            return node == goal
        elif depth > 0:
            self.metrics['nodes_expanded'] += 1
            self._update_branching_factor(node)
            self.visited.add(node)
            
            for neighbor in self.graph.neighbors(node):
                if neighbor not in self.visited:
                    self.parent[neighbor] = node
                    self.metrics['nodes_generated'] += 1
                    self.metrics['max_frontier_size'] = max(self.metrics['max_frontier_size'], 
                                                          len([n for n in self.graph.neighbors(node) 
                                                               if n not in self.visited]))
                    if self._depth_limited_search(neighbor, goal, depth - 1):
                        return True
        return False

class BestFirst(SearchAlgorithm):
    def __init__(self, graph: nx.Graph, coordinates: Dict[str, Tuple[float, float]], 
                 heuristic: str = 'euclidean'):
        super().__init__(graph, coordinates)
        self.heuristic = self._get_heuristic(heuristic)
    
    def _get_heuristic(self, name: str):
        def euclidean(node1: str, node2: str) -> float:
            if node1 not in self.coordinates or node2 not in self.coordinates:
                return 0.0
            x1, y1 = self.coordinates[node1]
            x2, y2 = self.coordinates[node2]
            return ((x2 - x1)**2 + (y2 - y1)**2)**0.5
        
        def manhattan(node1: str, node2: str) -> float:
            if node1 not in self.coordinates or node2 not in self.coordinates:
                return 0.0
            x1, y1 = self.coordinates[node1]
            x2, y2 = self.coordinates[node2]
            return abs(x2 - x1) + abs(y2 - y1)
        
        def zero(node1: str, node2: str) -> float:
            return 0.0
            
        heuristics = {'euclidean': euclidean, 'manhattan': manhattan, 'zero': zero}
        return heuristics.get(name, euclidean)
    
    def search(self, start: str, goal: str) -> Tuple[List[str], Dict[str, Any]]:
        start_time = time.perf_counter()
        self.visited = set()
        self.parent = {}
        self.metrics = {k: 0 for k in self.metrics.keys()}
        self.metrics['branching_factors'] = []
        
        frontier = []
        heapq.heappush(frontier, (self.heuristic(start, goal), start))
        self.parent[start] = None
        self.visited.add(start)
        self.metrics['nodes_generated'] = 1
        
        while frontier:
            self.metrics['max_frontier_size'] = max(self.metrics['max_frontier_size'], len(frontier))
            _, current = heapq.heappop(frontier)
            self.metrics['nodes_expanded'] += 1
            self._update_branching_factor(current)
            
            if current == goal:
                self.path = self._reconstruct_path(start, goal)
                self.metrics['path_cost'] = self._calculate_path_cost(self.path)
                self.metrics['solution_depth'] = len(self.path) - 1 if self.path else 0
                self.metrics['max_explored_size'] = len(self.visited)
                self.metrics['execution_time'] = time.perf_counter() - start_time
                return self.path, self.metrics
            
            for neighbor in self.graph.neighbors(current):
                if neighbor not in self.visited:
                    self.visited.add(neighbor)
                    self.parent[neighbor] = current
                    heuristic_val = self.heuristic(neighbor, goal)
                    heapq.heappush(frontier, (heuristic_val, neighbor))
                    self.metrics['nodes_generated'] += 1
        
        self.metrics['execution_time'] = time.perf_counter() - start_time
        return [], self.metrics

class AStar(SearchAlgorithm):
    def __init__(self, graph: nx.Graph, coordinates: Dict[str, Tuple[float, float]], 
                 heuristic: str = 'euclidean'):
        super().__init__(graph, coordinates)
        self.heuristic = self._get_heuristic(heuristic)
        self.g_score = {}
    
    def _get_heuristic(self, name: str):
        # Same implementation as BestFirst
        def euclidean(node1: str, node2: str) -> float:
            if node1 not in self.coordinates or node2 not in self.coordinates:
                return 0.0
            x1, y1 = self.coordinates[node1]
            x2, y2 = self.coordinates[node2]
            return ((x2 - x1)**2 + (y2 - y1)**2)**0.5
        
        def manhattan(node1: str, node2: str) -> float:
            if node1 not in self.coordinates or node2 not in self.coordinates:
                return 0.0
            x1, y1 = self.coordinates[node1]
            x2, y2 = self.coordinates[node2]
            return abs(x2 - x1) + abs(y2 - y1)
        
        def zero(node1: str, node2: str) -> float:
            return 0.0
            
        heuristics = {'euclidean': euclidean, 'manhattan': manhattan, 'zero': zero}
        return heuristics.get(name, euclidean)
    
    def search(self, start: str, goal: str) -> Tuple[List[str], Dict[str, Any]]:
        start_time = time.perf_counter()
        self.visited = set()
        self.parent = {}
        self.metrics = {k: 0 for k in self.metrics.keys()}
        self.metrics['branching_factors'] = []
        self.g_score = {}
        
        frontier = []
        self.g_score[start] = 0
        f_score = self.g_score[start] + self.heuristic(start, goal)
        heapq.heappush(frontier, (f_score, start))
        self.parent[start] = None
        self.metrics['nodes_generated'] = 1
        
        while frontier:
            self.metrics['max_frontier_size'] = max(self.metrics['max_frontier_size'], len(frontier))
            _, current = heapq.heappop(frontier)
            self.metrics['nodes_expanded'] += 1
            self._update_branching_factor(current)
            
            if current == goal:
                self.path = self._reconstruct_path(start, goal)
                self.metrics['path_cost'] = self._calculate_path_cost(self.path)
                self.metrics['solution_depth'] = len(self.path) - 1 if self.path else 0
                self.metrics['max_explored_size'] = len(self.visited)
                self.metrics['execution_time'] = time.perf_counter() - start_time
                return self.path, self.metrics
            
            self.visited.add(current)
            
            for neighbor in self.graph.neighbors(current):
                if neighbor in self.visited:
                    continue
                    
                edge_weight = self.graph[current][neighbor]['weight']
                tentative_g_score = self.g_score[current] + edge_weight
                
                if neighbor not in self.g_score or tentative_g_score < self.g_score[neighbor]:
                    self.parent[neighbor] = current
                    self.g_score[neighbor] = tentative_g_score
                    f_score = tentative_g_score + self.heuristic(neighbor, goal)
                    heapq.heappush(frontier, (f_score, neighbor))
                    self.metrics['nodes_generated'] += 1
        
        self.metrics['execution_time'] = time.perf_counter() - start_time
        return [], self.metrics