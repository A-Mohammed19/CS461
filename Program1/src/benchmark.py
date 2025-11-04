import time
import json
import csv
from typing import Dict, List, Any, Callable
from .metrics import MetricsCollector
import statistics

class BenchmarkHarness:
    def __init__(self):
        self.results = {}
        self.collector = MetricsCollector()
        
    def single_run(self, algorithm_func: Callable, algorithm_name: str,
                  start: str, goal: str, graph_info: str = "unknown") -> Dict[str, Any]:
        """Run single benchmark of an algorithm"""
        self.collector.start_collection()
        
        # Run algorithm
        path, algo_metrics = algorithm_func(start, goal)
        
        # Collect performance metrics
        perf_metrics = self.collector.stop_collection()
        self.collector.update_algorithm_metrics(algo_metrics)
        
        # Combine all metrics
        all_metrics = self.collector.metrics.copy()
        all_metrics.update({
            'algorithm': algorithm_name,
            'graph': graph_info,
            'start': start,
            'goal': goal,
            'path_found': len(path) > 0,
            'path_length': len(path) if path else 0,
            'optimal': True  # Will be compared against baseline later
        })
        
        return all_metrics
    
    def batch_run(self, algorithms: Dict[str, Callable], test_cases: List[Dict],
                 num_repeats: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """Run batch benchmarks across multiple algorithms and test cases"""
        all_results = {}
        
        for algo_name, algo_func in algorithms.items():
            algo_results = []
            
            for test_case in test_cases:
                for run in range(num_repeats):
                    metrics = self.single_run(
                        algo_func, algo_name,
                        test_case['start'], test_case['goal'],
                        test_case.get('graph', 'unknown')
                    )
                    metrics['run_id'] = run + 1
                    metrics['test_case'] = test_case.get('name', 'unknown')
                    metrics['seed'] = test_case.get('seed', run)
                    algo_results.append(metrics)
            
            all_results[algo_name] = algo_results
        
        return all_results
    
    def export_to_csv(self, results: Dict[str, List[Dict[str, Any]]], filename: str):
        """Export results to CSV file"""
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = [
                'algorithm', 'test_case', 'run_id', 'seed', 'path_found', 'path_length',
                'path_cost', 'nodes_expanded', 'nodes_generated', 'max_frontier_size',
                'wall_clock_time', 'memory_peak_rss', 'algorithmic_memory_footprint',
                'avg_branching_factor', 'solution_depth'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for algo_results in results.values():
                for result in algo_results:
                    writer.writerow({field: result.get(field, '') for field in fieldnames})
    
    def export_to_json(self, results: Dict[str, List[Dict[str, Any]]], filename: str):
        """Export results to JSON file"""
        with open(filename, 'w') as jsonfile:
            json.dump(results, jsonfile, indent=2)
    
    def generate_statistics(self, results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """Generate statistical summary of results"""
        stats = {}
        
        for algo_name, algo_results in results.items():
            # Filter successful runs
            successful_runs = [r for r in algo_results if r['path_found']]
            
            if successful_runs:
                stats[algo_name] = {
                    'success_rate': len(successful_runs) / len(algo_results),
                    'avg_time': statistics.mean([r['wall_clock_time'] for r in successful_runs]),
                    'std_time': statistics.stdev([r['wall_clock_time'] for r in successful_runs]) if len(successful_runs) > 1 else 0,
                    'avg_nodes_expanded': statistics.mean([r['nodes_expanded'] for r in successful_runs]),
                    'avg_path_cost': statistics.mean([r['path_cost'] for r in successful_runs]),
                    'avg_memory': statistics.mean([r['memory_peak_rss'] for r in successful_runs]),
                }
        
        return stats