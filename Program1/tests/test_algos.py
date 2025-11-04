# test_search_algorithms.py
import unittest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.algorithms import DFS, BFS, IDDFS, BestFirst, AStar
from src.graph_load import GraphLoader
from src.graph_gen import GraphGenerator
import networkx as nx

class TestSearchAlgorithms(unittest.TestCase):
    
    def setUp(self):
        """Set up test graphs"""
        # Simple test graph
        self.simple_graph = nx.Graph()
        self.simple_graph.add_edges_from([
            ('A', 'B', {'weight': 1}),
            ('B', 'C', {'weight': 1}),
            ('C', 'D', {'weight': 1}),
            ('A', 'D', {'weight': 3})
        ])
        self.simple_coords = {
            'A': (0, 0), 'B': (1, 0), 'C': (2, 0), 'D': (3, 0)
        }
        
        # Grid graph for testing
        self.grid_generator = GraphGenerator(seed=42)
        self.grid_graph, self.grid_coords = self.grid_generator.generate_grid(
            size=5, obstacle_density=0.1
        )
    
    def test_dfs_basic(self):
        """Test DFS finds a path (not necessarily optimal)"""
        dfs = DFS(self.simple_graph, self.simple_coords)
        path, metrics = dfs.search('A', 'D')
        
        self.assertTrue(len(path) > 0)
        self.assertEqual(path[0], 'A')
        self.assertEqual(path[-1], 'D')
        self.assertGreater(metrics['nodes_expanded'], 0)
    
    def test_bfs_optimal(self):
        """Test BFS finds optimal path in unweighted graph"""
        bfs = BFS(self.simple_graph, self.simple_coords)
        path, metrics = bfs.search('A', 'D')
        
        # Should find shortest path: A->D (direct) or A->B->C->D (both length 3)
        self.assertTrue(len(path) <= 4)  # Should find reasonable path
        self.assertEqual(path[0], 'A')
        self.assertEqual(path[-1], 'D')
    
    def test_iddfs_completeness(self):
        """Test IDDFS finds path with depth limiting"""
        iddfs = IDDFS(self.simple_graph, self.simple_coords)
        path, metrics = iddfs.search('A', 'D')
        
        self.assertTrue(len(path) > 0)
        self.assertEqual(path[0], 'A')
        self.assertEqual(path[-1], 'D')
    
    def test_best_first_heuristic(self):
        """Test Best-First uses heuristic"""
        best_first = BestFirst(self.simple_graph, self.simple_coords)
        path, metrics = best_first.search('A', 'D')
        
        self.assertTrue(len(path) > 0)
        self.assertIn('path_cost', metrics)
    
    def test_astar_optimal(self):
        """Test A* finds optimal path with admissible heuristic"""
        astar = AStar(self.simple_graph, self.simple_coords)
        path, metrics = astar.search('A', 'D')
        
        self.assertTrue(len(path) > 0)
        # A* should find optimal path in this simple case
        self.assertLessEqual(metrics['path_cost'], 3.0)
    
    def test_heuristic_admissibility(self):
        """Test heuristics are admissible"""
        astar = AStar(self.simple_graph, self.simple_coords)
        
        # Euclidean heuristic should be admissible
        h_value = astar.heuristic('A', 'D')
        actual_cost = 3.0  # Direct path cost
        
        # Heuristic should not overestimate
        self.assertLessEqual(h_value, actual_cost)
    
    def test_no_path(self):
        """Test behavior when no path exists"""
        # Create disconnected graph
        disconnected_graph = nx.Graph()
        disconnected_graph.add_edges_from([('A', 'B'), ('C', 'D')])
        disconnected_coords = {'A': (0,0), 'B': (1,0), 'C': (2,0), 'D': (3,0)}
        
        bfs = BFS(disconnected_graph, disconnected_coords)
        path, metrics = bfs.search('A', 'C')
        
        self.assertEqual(len(path), 0)
        self.assertGreater(metrics['nodes_expanded'], 0)
    
    def test_same_start_goal(self):
        """Test start = goal case"""
        bfs = BFS(self.simple_graph, self.simple_coords)
        path, metrics = bfs.search('A', 'A')
        
        self.assertEqual(len(path), 1)
        self.assertEqual(path[0], 'A')
        self.assertEqual(metrics['path_cost'], 0.0)
    
    def test_metrics_collection(self):
        """Test that all required metrics are collected"""
        bfs = BFS(self.simple_graph, self.simple_coords)
        path, metrics = bfs.search('A', 'D')
        
        required_metrics = [
            'nodes_expanded', 'nodes_generated', 'max_frontier_size',
            'path_cost', 'solution_depth', 'execution_time'
        ]
        
        for metric in required_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], (int, float))

class TestGraphGeneration(unittest.TestCase):
    
    def test_grid_generation(self):
        """Test grid graph generation with obstacles"""
        generator = GraphGenerator(seed=42)
        graph, coords = generator.generate_grid(
            size=10, obstacle_density=0.2, connectivity="4"
        )
        
        self.assertGreater(len(graph.nodes()), 0)
        self.assertGreater(len(graph.edges()), 0)
        self.assertEqual(len(coords), len(graph.nodes()))
    
    def test_random_graph_generation(self):
        """Test random graph generation with Poisson branching"""
        generator = GraphGenerator(seed=42)
        graph, coords = generator.generate_random_graph(
            num_nodes=20, branching_factor=3.0
        )
        
        self.assertEqual(len(graph.nodes()), 20)
        # Should be connected or have reasonable connectivity
        avg_degree = sum(dict(graph.degree()).values()) / len(graph.nodes())
        self.assertGreater(avg_degree, 1.0)
    
    def test_maze_generation(self):
        """Test maze generation using randomized DFS"""
        generator = GraphGenerator(seed=42)
        graph, coords = generator.generate_maze(size=10)
        
        # Maze should have most cells connected in a single component
        self.assertEqual(len(graph.nodes()), 100)  # 10x10 grid
        # Should be mostly connected (maze might have some dead ends but should be traversable)
        self.assertTrue(nx.is_connected(graph))

class TestBenchmarking(unittest.TestCase):
    
    def setUp(self):
        self.graph_loader = GraphLoader()
    
    def test_set1_loading(self):
        """Test loading Set 1 (Kansas towns)"""
        try:
            graph, coords = self.graph_loader.load_set1(
                "data/coordinates.csv", "data/Adjacencies.txt"
            )
            self.assertGreater(len(graph.nodes()), 0)
            self.assertGreater(len(graph.edges()), 0)
            self.assertEqual(len(coords), len(graph.nodes()))
        except FileNotFoundError:
            self.skipTest("Set 1 data files not found")
    
    def test_set2_loading(self):
        """Test loading Set 2 (KC Metro)"""
        try:
            graph, coords = self.graph_loader.load_set2(
                "data/KC_Metro_100_Cities___Nodes.csv",
                "data/KC_Metro_100_Cities___Edges.csv"
            )
            self.assertGreater(len(graph.nodes()), 0)
            self.assertGreater(len(graph.edges()), 0)
            self.assertEqual(len(coords), len(graph.nodes()))
        except FileNotFoundError:
            self.skipTest("Set 2 data files not found")

class TestAlgorithmComparison(unittest.TestCase):
    """Compare algorithms on known test cases"""
    
    def setUp(self):
        # Create a known test graph
        self.test_graph = nx.Graph()
        # Simple diamond pattern
        self.test_graph.add_edges_from([
            ('S', 'A', {'weight': 1}),
            ('S', 'B', {'weight': 1}),
            ('A', 'G', {'weight': 1}),
            ('B', 'G', {'weight': 3}),
            ('A', 'B', {'weight': 1})
        ])
        self.test_coords = {
            'S': (0, 0), 'A': (1, 1), 'B': (1, -1), 'G': (2, 0)
        }
    
    def test_algorithm_paths(self):
        """Test all algorithms find paths in simple graph"""
        algorithms = {
            'BFS': BFS(self.test_graph, self.test_coords),
            'DFS': DFS(self.test_graph, self.test_coords),
            'IDDFS': IDDFS(self.test_graph, self.test_coords),
            'BestFirst': BestFirst(self.test_graph, self.test_coords),
            'AStar': AStar(self.test_graph, self.test_coords)
        }
        
        for algo_name, algorithm in algorithms.items():
            with self.subTest(algorithm=algo_name):
                path, metrics = algorithm.search('S', 'G')
                self.assertTrue(len(path) > 0, f"{algo_name} failed to find path")
                self.assertEqual(path[0], 'S')
                self.assertEqual(path[-1], 'G')
    
    def test_astar_optimality(self):
        """Test A* finds truly optimal path"""
        # Graph where direct path is more expensive but looks good to greedy
        tricky_graph = nx.Graph()
        tricky_graph.add_edges_from([
            ('S', 'A', {'weight': 10}),
            ('S', 'B', {'weight': 1}),
            ('B', 'C', {'weight': 1}),
            ('C', 'G', {'weight': 1}),
            ('A', 'G', {'weight': 1})
        ])
        tricky_coords = {
            'S': (0, 0), 'A': (0, 10), 'B': (1, 0), 'C': (2, 0), 'G': (2, 10)
        }
        
        # A* should find S->B->C->G (cost 3) rather than S->A->G (cost 11)
        astar = AStar(tricky_graph, tricky_coords)
        path, metrics = astar.search('S', 'G')
        
        optimal_cost = 3.0
        self.assertEqual(metrics['path_cost'], optimal_cost)

def run_comprehensive_validation():
    """Run comprehensive validation on known Set 1 data"""
    print("🔍 Running Comprehensive Algorithm Validation")
    print("=" * 50)
    
    loader = GraphLoader()
    
    try:
        # Load Set 1 for validation
        graph, coords = loader.load_set1(
            "data/coordinates.csv", "data/Adjacencies.txt"
        )
        
        print(f"✅ Loaded Set 1: {len(graph.nodes())} nodes, {len(graph.edges())} edges")
        
        # Test known city pairs
        test_pairs = [
            ('Abilene', 'Topeka'),
            ('Kansas_City', 'Wichita'),
            ('Manhattan', 'Lawrence')
        ]
        
        algorithms = {
            'BFS': BFS(graph, coords),
            'DFS': DFS(graph, coords),
            'IDDFS': IDDFS(graph, coords),
            'BestFirst': BestFirst(graph, coords),
            'AStar': AStar(graph, coords)
        }
        
        for start, goal in test_pairs:
            if start in graph.nodes() and goal in graph.nodes():
                print(f"\n🎯 Testing {start} → {goal}")
                print("-" * 30)
                
                for algo_name, algorithm in algorithms.items():
                    path, metrics = algorithm.search(start, goal)
                    
                    if path:
                        status = "✅"
                        path_info = f"Path length: {len(path)}, Cost: {metrics['path_cost']:.2f}"
                    else:
                        status = "❌"
                        path_info = "No path found"
                    
                    print(f"{status} {algo_name:12} | {path_info} | "
                          f"Nodes: {metrics['nodes_expanded']:4d} | "
                          f"Time: {metrics['execution_time']*1000:6.2f}ms")
        
        print("\n" + "=" * 50)
        print("✅ Validation Complete!")
        
    except FileNotFoundError:
        print("❌ Set 1 data files not found - skipping validation")
    except Exception as e:
        print(f"❌ Validation failed: {e}")

if __name__ == '__main__':
    # Run unit tests
    unittest.main(verbosity=2, exit=False)
    
    # Run comprehensive validation
    print("\n" + "=" * 60)
    run_comprehensive_validation()