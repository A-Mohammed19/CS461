# src/metrics.py
import time
import tracemalloc
import psutil
import os
from typing import Dict, List, Any, Callable
import statistics

class MetricsCollector:
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.metrics = {}
        self.start_time = 0
        self.start_memory = 0
        
    def start_collection(self):
        """Start collecting performance metrics"""
        self.start_time = time.perf_counter()
        tracemalloc.start()
        self.start_memory = self.process.memory_info().rss
        
    def stop_collection(self):
        """Stop collection and return all metrics"""
        end_time = time.perf_counter()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        end_memory = self.process.memory_info().rss
        
        self.metrics['wall_clock_time'] = end_time - self.start_time
        self.metrics['memory_peak_tracemalloc'] = peak / 1024  # KB
        self.metrics['memory_peak_rss'] = (end_memory - self.start_memory) / 1024  # KB
        
        return self.metrics
    
    def update_algorithm_metrics(self, algo_metrics: Dict[str, Any]):
        """Update with algorithm-specific metrics"""
        self.metrics.update(algo_metrics)
        
        # Calculate branching factor statistics
        if 'branching_factors' in algo_metrics and algo_metrics['branching_factors']:
            factors = algo_metrics['branching_factors']
            self.metrics['avg_branching_factor'] = statistics.mean(factors)
            self.metrics['max_branching_factor'] = max(factors)
        else:
            self.metrics['avg_branching_factor'] = 0
            self.metrics['max_branching_factor'] = 0
            
        # Calculate algorithmic memory footprint
        frontier_size = algo_metrics.get('max_frontier_size', 0)
        explored_size = algo_metrics.get('max_explored_size', 0)
        self.metrics['algorithmic_memory_footprint'] = frontier_size + explored_size