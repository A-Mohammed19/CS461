import pandas as pd
import networkx as nx
from typing import Dict, Tuple, List

class GraphLoader:
    def __init__(self):
        self.graph = nx.Graph()
        self.coordinates = {}
        
    def load_set1(self, coord_file: str, adj_file: str) -> Tuple[nx.Graph, Dict]:
        """Load Set 1 (Kansas towns) with symmetric adjacency"""
        self.graph = nx.Graph()
        self.coordinates = {}
        
        # Load coordinates
        coords_df = pd.read_csv(coord_file, header=None, names=['city', 'lat', 'lon'])
        for _, row in coords_df.iterrows():
            city = row['city'].strip()
            self.coordinates[city] = (float(row['lat']), float(row['lon']))
            self.graph.add_node(city, pos=(float(row['lon']), float(row['lat'])))
        
        # Load adjacencies and ensure symmetry
        with open(adj_file, 'r') as f:
            for line in f:
                cities = line.strip().split()
                if len(cities) >= 2:
                    city1, city2 = cities[0], cities[1]
                    if city1 in self.coordinates and city2 in self.coordinates:
                        distance = self._calculate_distance(city1, city2)
                        self.graph.add_edge(city1, city2, weight=distance)
                        self.graph.add_edge(city2, city1, weight=distance)  # Ensure symmetry
        
        return self.graph, self.coordinates
    
    def load_set2(self, nodes_file: str, edges_file: str) -> Tuple[nx.Graph, Dict]:
        """Load Set 2 (KC Metro) with symmetric adjacency"""
        self.graph = nx.Graph()
        self.coordinates = {}
        
        # Load nodes
        nodes_df = pd.read_csv(nodes_file)
        for _, row in nodes_df.iterrows():
            city = row['city']
            self.coordinates[city] = (float(row['lat']), float(row['lon']))
            self.graph.add_node(city, pos=(float(row['lon']), float(row['lat'])))
        
        # Load edges and ensure symmetry
        edges_df = pd.read_csv(edges_file)
        for _, row in edges_df.iterrows():
            from_city, to_city = row['from'], row['to']
            if from_city in self.coordinates and to_city in self.coordinates:
                distance = self._calculate_distance(from_city, to_city)
                self.graph.add_edge(from_city, to_city, weight=distance, road=row['road'])
                self.graph.add_edge(to_city, from_city, weight=distance, road=row['road'])  # Ensure symmetry
        
        return self.graph, self.coordinates
    
    def _calculate_distance(self, city1: str, city2: str) -> float:
        """Calculate Euclidean distance between cities"""
        x1, y1 = self.coordinates[city1]
        x2, y2 = self.coordinates[city2]
        return ((x2 - x1)**2 + (y2 - y1)**2)**0.5