import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib
matplotlib.use('TkAgg')
import networkx as nx
import threading
import random
import math
from typing import Dict, List, Tuple, Any
from collections import deque
import sys
import os
from collections import deque
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.algorithms import DFS, BFS, IDDFS, BestFirst, AStar
from src.graph_load import GraphLoader
from src.benchmark import BenchmarkHarness

class GraphGenerator:
    """Graph generation functionality"""
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
        
        nodes = []
        for x in range(size):
            for y in range(size):
                if random.random() > obstacle_density:
                    node_id = f"({x},{y})"
                    self.coordinates[node_id] = (x, y)
                    self.graph.add_node(node_id, pos=(x, y))
                    nodes.append((x, y, node_id))
        
        for i, (x1, y1, node1) in enumerate(nodes):
            for j, (x2, y2, node2) in enumerate(nodes[i+1:], i+1):
                if connectivity == "4":
                    if (abs(x1 - x2) == 1 and y1 == y2) or (abs(y1 - y2) == 1 and x1 == x2):
                        weight = random.uniform(1, 10) if weighted else 1.0
                        self.graph.add_edge(node1, node2, weight=weight)
                else:
                    if abs(x1 - x2) <= 1 and abs(y1 - y2) <= 1 and (x1 != x2 or y1 != y2):
                        weight = random.uniform(1, 10) if weighted else math.sqrt((x1-x2)**2 + (y1-y2)**2)
                        self.graph.add_edge(node1, node2, weight=weight)
        
        return self.graph, self.coordinates
    
    def generate_random_graph(self, num_nodes: int, branching_factor: float,
                            weight_range: Tuple[float, float] = (1, 10),
                            seed: int = None) -> Tuple[nx.Graph, Dict]:
        """Generate random graph"""
        if seed is not None:
            random.seed(seed)
            
        self.graph = nx.Graph()
        self.coordinates = {}
        
        for i in range(num_nodes):
            node_id = f"Node_{i}"
            x, y = random.uniform(0, 100), random.uniform(0, 100)
            self.coordinates[node_id] = (x, y)
            self.graph.add_node(node_id, pos=(x, y))
        
        nodes = list(self.graph.nodes())
        for node in nodes:
            num_connections = int(random.gauss(branching_factor, branching_factor/3))
            num_connections = max(1, min(num_connections, num_nodes - 1))
            
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
        
        grid = {}
        for x in range(size):
            for y in range(size):
                node_id = f"({x},{y})"
                self.coordinates[node_id] = (x, y)
                self.graph.add_node(node_id, pos=(x, y))
                grid[(x, y)] = node_id
        
        visited = set()
        stack = []
        start = (0, 0)
        visited.add(start)
        stack.append(start)
        
        while stack:
            current = stack[-1]
            x, y = current
            
            neighbors = []
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx_, ny = x + dx, y + dy
                if 0 <= nx_ < size and 0 <= ny < size and (nx_, ny) not in visited:
                    neighbors.append((nx_, ny))
            
            if neighbors:
                next_cell = random.choice(neighbors)
                self.graph.add_edge(grid[current], grid[next_cell], weight=1.0)
                visited.add(next_cell)
                stack.append(next_cell)
            else:
                stack.pop()
        
        return self.graph, self.coordinates

class SearchVisualizationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🔍 AI Search Algorithms Lab")
        self.root.geometry("1400x900")
        
        # Data management
        self.graph_loader = GraphLoader()
        self.graph_generator = GraphGenerator()
        self.benchmark_harness = BenchmarkHarness()
        self.graph = None
        self.coordinates = {}
        self.current_graph_type = ""
        
        # Animation state
        self.animation_frames = []
        self.current_frame = 0
        self.is_animating = False
        self.animation_job = None
        self.animation_delay = 500
        
        # Color scheme
        self.colors = {
            'start': '#10B981',
            'goal': '#EF4444',
            'frontier': '#FBBF24',
            'explored': '#60A5FA',
            'current': '#8B5CF6',
            'path': '#EC4899',
            'default': '#E5E7EB',
            'obstacle': '#374151',
            'edge': '#9CA3AF',
            'text': '#1F2937',
            'bg': '#FFFFFF',
            'panel_bg': '#F9FAFB',
            'accent': '#3B82F6'
        }
        
        self.setup_styles()
        self.setup_gui()
    
    def setup_styles(self):
        """Configure ttk styles"""
        style = ttk.Style()
        try:
            style.theme_use('clam')
        except:
            pass
        
        style.configure('TFrame', background=self.colors['panel_bg'])
        style.configure('TLabel', background=self.colors['panel_bg'], 
                       foreground=self.colors['text'], font=('Segoe UI', 9))
        style.configure('TLabelframe', background=self.colors['panel_bg'])
        style.configure('TLabelframe.Label', background=self.colors['panel_bg'], 
                       foreground=self.colors['text'], font=('Segoe UI', 9, 'bold'))
        style.configure('Accent.TButton', font=('Segoe UI', 9, 'bold'))
        style.configure('Success.TButton', font=('Segoe UI', 9, 'bold'))
        style.configure('Danger.TButton', font=('Segoe UI', 9, 'bold'))
    
    def setup_gui(self):
        """Setup complete GUI layout"""
        main_pane = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        control_frame = ttk.Frame(main_pane, style='TFrame')
        main_pane.add(control_frame, weight=1)
        
        viz_frame = ttk.Frame(main_pane, style='TFrame')
        main_pane.add(viz_frame, weight=2)
        
        self.setup_control_panel(control_frame)
        self.setup_visualization_panel(viz_frame)
    
    def setup_control_panel(self, parent):
        """Setup control panel with tabs"""
        notebook = ttk.Notebook(parent)
        notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        graph_frame = ttk.Frame(notebook, padding="10")
        notebook.add(graph_frame, text="📊 Graph")
        self.setup_graph_tab(graph_frame)
        
        algo_frame = ttk.Frame(notebook, padding="10")
        notebook.add(algo_frame, text="⚡ Algorithm")
        self.setup_algorithm_tab(algo_frame)
        
        bench_frame = ttk.Frame(notebook, padding="10")
        notebook.add(bench_frame, text="📈 Benchmark")
        self.setup_benchmark_tab(bench_frame)
        
        metrics_frame = ttk.Frame(notebook, padding="10")
        notebook.add(metrics_frame, text="📋 Metrics")
        self.setup_metrics_tab(metrics_frame)
    
    def setup_graph_tab(self, parent):
        """Setup graph selection and loading"""
        preset_frame = ttk.LabelFrame(parent, text="🗂️ Preset Graphs", padding="10")
        preset_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(preset_frame, text="🏙️ Load Set 1 (Kansas Towns)", 
                  command=self.load_set1, style='Accent.TButton').pack(fill=tk.X, pady=2)
        ttk.Button(preset_frame, text="🏢 Load Set 2 (KC Metro)", 
                  command=self.load_set2, style='Accent.TButton').pack(fill=tk.X, pady=2)
        
        random_frame = ttk.LabelFrame(parent, text="🎲 Generated Graphs", padding="10")
        random_frame.pack(fill=tk.X, pady=5)
        
        grid_frame = ttk.Frame(random_frame)
        grid_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(grid_frame, text="Size:").pack(side=tk.LEFT)
        self.grid_size_var = tk.IntVar(value=10)
        ttk.Spinbox(grid_frame, from_=5, to=50, textvariable=self.grid_size_var, width=5).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(grid_frame, text="Obstacle %:").pack(side=tk.LEFT, padx=(10,0))
        self.obstacle_var = tk.DoubleVar(value=0.2)
        ttk.Spinbox(grid_frame, from_=0, to=0.8, increment=0.1, textvariable=self.obstacle_var, width=5).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(random_frame, text="🔲 Generate Grid World", 
                  command=self.generate_grid, style='Success.TButton').pack(fill=tk.X, pady=2)
        
        random_graph_frame = ttk.Frame(random_frame)
        random_graph_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(random_graph_frame, text="Nodes:").pack(side=tk.LEFT)
        self.random_nodes_var = tk.IntVar(value=30)
        ttk.Spinbox(random_graph_frame, from_=10, to=100, textvariable=self.random_nodes_var, width=5).pack(side=tk.LEFT, padx=5)
        
        ttk.Label(random_graph_frame, text="Branching:").pack(side=tk.LEFT, padx=(10,0))
        self.branching_var = tk.DoubleVar(value=3.0)
        ttk.Spinbox(random_graph_frame, from_=1, to=10, increment=0.5, textvariable=self.branching_var, width=5).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(random_frame, text="🕸️ Generate Random Graph", 
                  command=self.generate_random, style='Success.TButton').pack(fill=tk.X, pady=2)
        
        ttk.Button(random_frame, text="🧩 Generate Maze", 
                  command=self.generate_maze, style='Success.TButton').pack(fill=tk.X, pady=2)
        
        info_frame = ttk.LabelFrame(parent, text="ℹ️ Graph Information", padding="10")
        info_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.graph_info_text = tk.Text(info_frame, height=8, font=("Consolas", 9), 
                                       bg=self.colors['bg'], fg=self.colors['text'],
                                       relief='solid', bd=1, padx=5, pady=5, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(info_frame, orient=tk.VERTICAL, command=self.graph_info_text.yview)
        self.graph_info_text.configure(yscrollcommand=scrollbar.set)
        self.graph_info_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.graph_info_text.insert(tk.END, "📊 No graph loaded.\n\n")
        self.graph_info_text.insert(tk.END, "Please load a preset graph or generate a random graph to begin.\n\n")
        self.graph_info_text.insert(tk.END, "Available options:\n")
        self.graph_info_text.insert(tk.END, "  • Set 1: Kansas Towns (46 cities)\n")
        self.graph_info_text.insert(tk.END, "  • Set 2: KC Metro (100 cities)\n")
        self.graph_info_text.insert(tk.END, "  • Grid World (customizable)\n")
        self.graph_info_text.insert(tk.END, "  • Random Graph (customizable)\n")
        self.graph_info_text.insert(tk.END, "  • Maze (15x15)")
    
    def setup_algorithm_tab(self, parent):
        """Setup algorithm selection and controls"""
        node_frame = ttk.LabelFrame(parent, text="🎯 Start & Goal Selection", padding="10")
        node_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(node_frame, text="Start Node:").pack(anchor=tk.W)
        self.start_var = tk.StringVar()
        self.start_combo = ttk.Combobox(node_frame, textvariable=self.start_var, state="readonly")
        self.start_combo.pack(fill=tk.X, pady=2)
        self.start_combo.bind('<<ComboboxSelected>>', lambda e: self.update_visualization_on_node_change())
        
        ttk.Label(node_frame, text="Goal Node:").pack(anchor=tk.W, pady=(10,2))
        self.goal_var = tk.StringVar()
        self.goal_combo = ttk.Combobox(node_frame, textvariable=self.goal_var, state="readonly")
        self.goal_combo.pack(fill=tk.X, pady=2)
        self.goal_combo.bind('<<ComboboxSelected>>', lambda e: self.update_visualization_on_node_change())
        
        algo_frame = ttk.LabelFrame(parent, text="🔍 Search Algorithm", padding="10")
        algo_frame.pack(fill=tk.X, pady=5)
        
        self.algo_var = tk.StringVar(value="BFS")
        algorithms = [
            ("🔷 Breadth-First Search (BFS)", "BFS"), 
            ("🔶 Depth-First Search (DFS)", "DFS"),
            ("🔺 Iterative Deepening DFS (IDDFS)", "IDDFS"),
            ("⭐ Greedy Best-First Search", "BestFirst"),
            ("🚀 A* Search", "AStar")
        ]
        
        for text, value in algorithms:
            ttk.Radiobutton(algo_frame, text=text, variable=self.algo_var, 
                           value=value).pack(anchor=tk.W, pady=2)
        
        heuristic_frame = ttk.Frame(algo_frame)
        heuristic_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(heuristic_frame, text="Heuristic:").pack(side=tk.LEFT)
        self.heuristic_var = tk.StringVar(value="euclidean")
        heuristic_combo = ttk.Combobox(heuristic_frame, textvariable=self.heuristic_var,
                                      values=["euclidean", "manhattan", "zero"],
                                      state="readonly", width=12)
        heuristic_combo.pack(side=tk.LEFT, padx=5)
        
        control_frame = ttk.LabelFrame(parent, text="🎮 Search Controls", padding="10")
        control_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(control_frame, text="▶️ Run Search", 
                  command=self.run_search, style='Success.TButton').pack(fill=tk.X, pady=2)
        ttk.Button(control_frame, text="🔄 Reset Visualization", 
                  command=self.reset_visualization, style='Accent.TButton').pack(fill=tk.X, pady=2)
        
        anim_frame = ttk.LabelFrame(parent, text="🎥 Animation Controls", padding="10")
        anim_frame.pack(fill=tk.X, pady=5)
        
        button_frame = ttk.Frame(anim_frame)
        button_frame.pack(fill=tk.X, pady=2)
        
        ttk.Button(button_frame, text="⏮️", 
                  command=self.first_frame, style='Accent.TButton', width=3).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="⏪", 
                  command=self.prev_frame, style='Accent.TButton', width=3).pack(side=tk.LEFT, padx=2)
        self.play_button = ttk.Button(button_frame, text="▶️", 
                                     command=self.toggle_play, style='Accent.TButton', width=3)
        self.play_button.pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="⏩", 
                  command=self.next_frame, style='Accent.TButton', width=3).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="⏭️", 
                  command=self.last_frame, style='Accent.TButton', width=3).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="⏹️", 
                  command=self.stop_animation, style='Danger.TButton', width=3).pack(side=tk.LEFT, padx=2)
        
        speed_frame = ttk.Frame(anim_frame)
        speed_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(speed_frame, text="Speed:").pack(side=tk.LEFT)
        self.speed_var = tk.DoubleVar(value=1.0)
        speed_scale = ttk.Scale(speed_frame, from_=0.1, to=3.0, variable=self.speed_var,
                               orient=tk.HORIZONTAL, command=self.update_animation_speed)
        speed_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        self.speed_label = ttk.Label(speed_frame, text="1.0x")
        self.speed_label.pack(side=tk.RIGHT, padx=5)
        
        progress_frame = ttk.Frame(anim_frame)
        progress_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(progress_frame, text="Progress:").pack(side=tk.LEFT)
        self.progress_var = tk.IntVar(value=0)
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, 
                                           maximum=100, mode='determinate')
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        self.progress_text = ttk.Label(progress_frame, text="0/0")
        self.progress_text.pack(side=tk.RIGHT)
    
    def setup_benchmark_tab(self, parent):
        """Setup benchmarking controls"""
        single_frame = ttk.LabelFrame(parent, text="📊 Single Benchmark", padding="10")
        single_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(single_frame, text="🎯 Run Current Configuration", 
                  command=self.run_single_benchmark, style='Success.TButton').pack(fill=tk.X, pady=2)
        
        batch_frame = ttk.LabelFrame(parent, text="📈 Batch Benchmark", padding="10")
        batch_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(batch_frame, text="📊 Run All Algorithms Comparison", 
                  command=self.run_batch_benchmark, style='Accent.TButton').pack(fill=tk.X, pady=2)
        
        export_frame = ttk.LabelFrame(parent, text="💾 Export Results", padding="10")
        export_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(export_frame, text="📄 Export to CSV", 
                  command=self.export_results, style='Accent.TButton').pack(fill=tk.X, pady=2)
        
        results_frame = ttk.LabelFrame(parent, text="📋 Benchmark Results", padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        self.benchmark_text = tk.Text(results_frame, height=10, font=("Consolas", 9), 
                                     bg=self.colors['bg'], fg=self.colors['text'],
                                     relief='solid', bd=1, padx=5, pady=5, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.benchmark_text.yview)
        self.benchmark_text.configure(yscrollcommand=scrollbar.set)
        self.benchmark_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def setup_metrics_tab(self, parent):
        """Setup metrics display"""
        metrics_frame = ttk.LabelFrame(parent, text="📊 Search Metrics", padding="10")
        metrics_frame.pack(fill=tk.BOTH, expand=True)
        
        self.metrics_text = tk.Text(metrics_frame, height=15, font=("Consolas", 10), 
                                   bg=self.colors['bg'], fg=self.colors['text'],
                                   relief='solid', bd=1, padx=5, pady=5, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(metrics_frame, orient=tk.VERTICAL, command=self.metrics_text.yview)
        self.metrics_text.configure(yscrollcommand=scrollbar.set)
        
        self.metrics_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.metrics_text.insert(tk.END, "🔍 Search metrics will appear here after running an algorithm.\n\n")
        self.metrics_text.insert(tk.END, "📋 Available metrics:\n")
        self.metrics_text.insert(tk.END, "  • Path found (Y/N)\n")
        self.metrics_text.insert(tk.END, "  • Path length and cost\n")
        self.metrics_text.insert(tk.END, "  • Nodes expanded/generated\n")
        self.metrics_text.insert(tk.END, "  • Execution time\n")
        self.metrics_text.insert(tk.END, "  • Memory usage\n")
        self.metrics_text.insert(tk.END, "  • Branching factors\n")
        self.metrics_text.insert(tk.END, "  • Frontier size\n")
    
    def setup_visualization_panel(self, parent):
        """Setup matplotlib visualization"""
        plt.style.use('seaborn-v0_8-whitegrid')
        self.fig, self.ax = plt.subplots(figsize=(10, 8), facecolor=self.colors['panel_bg'])
        self.fig.patch.set_facecolor(self.colors['panel_bg'])
        self.ax.set_facecolor(self.colors['bg'])
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        legend_frame = ttk.LabelFrame(parent, text="🎨 Legend", padding="10")
        legend_frame.pack(fill=tk.X, pady=5)
        
        legend_text = (
            "🟢 Start Node  |  🔴 Goal Node  |  💗 Path  |  "
            "⬜ Unvisited  |  🔵 Explored  |  🟡 Frontier  |  🟣 Current"
        )
        ttk.Label(legend_frame, text=legend_text, font=('Segoe UI', 9)).pack()
        
        frontier_frame = ttk.LabelFrame(parent, text="🎯 Search State Display", padding="10")
        frontier_frame.pack(fill=tk.X, pady=5)
        
        self.frontier_text = tk.Text(frontier_frame, height=4, font=("Consolas", 8), 
                                    bg=self.colors['bg'], fg=self.colors['text'],
                                    relief='solid', bd=1, padx=5, pady=5, wrap=tk.WORD)
        self.frontier_text.pack(fill=tk.X)
        self.frontier_text.insert(tk.END, "Run a search to see frontier nodes and exploration progress...")
    
    def update_visualization_on_node_change(self):
        """Update visualization when start/goal nodes change"""
        if self.graph:
            self.draw_graph()
    
    def load_set1(self):
        """Load Set 1 dataset - Kansas Towns"""
        try:
            self.graph, self.coordinates = self.graph_loader.load_set1(
                "data/coordinates.csv", "data/Adjacencies.txt"
            )
            self.current_graph_type = "Set 1 - Kansas Towns"
            self.update_node_combos()
            self.draw_graph()
            self.update_graph_info()
            messagebox.showinfo("Success", "✅ Set 1 loaded successfully!\n\n46 Kansas towns with geographic coordinates.")
        except Exception as e:
            messagebox.showerror("Error", f"❌ Failed to load Set 1: {str(e)}\n\nPlease ensure data files are in the 'data/' directory.")
    
    def load_set2(self):
        """Load Set 2 dataset - KC Metro"""
        try:
            self.graph, self.coordinates = self.graph_loader.load_set2(
                "data/KC_Metro_100_Cities___Nodes.csv",
                "data/KC_Metro_100_Cities___Edges.csv"
            )
            self.current_graph_type = "Set 2 - KC Metro"
            self.update_node_combos()
            self.draw_graph()
            self.update_graph_info()
            messagebox.showinfo("Success", "✅ Set 2 loaded successfully!\n\n100 KC Metro cities with road connections.")
        except Exception as e:
            messagebox.showerror("Error", f"❌ Failed to load Set 2: {str(e)}\n\nPlease ensure data files are in the 'data/' directory.")
    
    def generate_grid(self):
        """Generate grid world"""
        try:
            size = self.grid_size_var.get()
            self.graph, self.coordinates = self.graph_generator.generate_grid(
                size=size,
                obstacle_density=self.obstacle_var.get(),
                connectivity="4",
                weighted=False,
                seed=42
            )
            self.current_graph_type = f"Grid World {size}x{size}"
            self.update_node_combos()
            self.draw_graph()
            self.update_graph_info()
            messagebox.showinfo("Success", f"✅ Grid world generated!\n\n{size}x{size} with {int(self.obstacle_var.get()*100)}% obstacles.")
        except Exception as e:
            messagebox.showerror("Error", f"❌ Failed to generate grid: {str(e)}")
    
    def generate_random(self):
        """Generate random graph"""
        try:
            nodes = self.random_nodes_var.get()
            self.graph, self.coordinates = self.graph_generator.generate_random_graph(
                num_nodes=nodes,
                branching_factor=self.branching_var.get(),
                seed=42
            )
            self.current_graph_type = f"Random Graph - {nodes} nodes"
            self.update_node_combos()
            self.draw_graph()
            self.update_graph_info()
            messagebox.showinfo("Success", f"✅ Random graph generated!\n\n{nodes} nodes with branching ~{self.branching_var.get()}.")
        except Exception as e:
            messagebox.showerror("Error", f"❌ Failed to generate random graph: {str(e)}")
    
    def generate_maze(self):
        """Generate maze"""
        try:
            self.graph, self.coordinates = self.graph_generator.generate_maze(
                size=15, seed=42
            )
            self.current_graph_type = "Maze 15x15"
            self.update_node_combos()
            self.draw_graph()
            self.update_graph_info()
            messagebox.showinfo("Success", "✅ Maze generated!\n\n15x15 maze using randomized DFS.")
        except Exception as e:
            messagebox.showerror("Error", f"❌ Failed to generate maze: {str(e)}")
    
    def update_node_combos(self):
        """Update node selection comboboxes"""
        if self.graph:
            nodes = sorted(list(self.graph.nodes()))
            self.start_combo['values'] = nodes
            self.goal_combo['values'] = nodes
            
            if nodes:
                self.start_var.set(nodes[0])
                self.goal_var.set(nodes[-1] if len(nodes) > 1 else nodes[0])
    
    def update_graph_info(self):
        """Update graph information display"""
        if self.graph:
            nodes_count = len(self.graph.nodes())
            edges_count = len(self.graph.edges())
            avg_degree = sum(dict(self.graph.degree()).values()) / nodes_count if nodes_count > 0 else 0
            is_connected = nx.is_connected(self.graph)
            
            info = f"📊 Graph Type: {self.current_graph_type}\n"
            info += f"{'='*50}\n\n"
            info += f"📍 Nodes: {nodes_count}\n"
            info += f"🔗 Edges: {edges_count}\n"
            info += f"📈 Avg Degree: {avg_degree:.2f}\n"
            info += f"🔄 Connected: {'Yes ✓' if is_connected else 'No ✗'}\n\n"
            
            if "Kansas" in self.current_graph_type or "KC Metro" in self.current_graph_type:
                nodes_list = sorted(list(self.graph.nodes()))
                info += f"📋 Sample Cities (first 10):\n"
                for i, city in enumerate(nodes_list[:10], 1):
                    info += f"  {i}. {city}\n"
                if len(nodes_list) > 10:
                    info += f"  ... and {len(nodes_list) - 10} more\n"
            
            self.graph_info_text.delete(1.0, tk.END)
            self.graph_info_text.insert(1.0, info)
    
    def draw_graph(self, highlight_path=None):
        """Draw the graph with GUARANTEED visible city names for Kansas/KC Metro"""
        if not self.graph:
            self.ax.clear()
            self.ax.set_facecolor(self.colors['bg'])
            self.ax.text(0.5, 0.5, '🔍 Load a graph to begin', 
                        ha='center', va='center', fontsize=16, 
                        transform=self.ax.transAxes, color=self.colors['text'])
            self.ax.axis('off')
            self.canvas.draw()
            return
            
        self.ax.clear()
        self.ax.set_facecolor(self.colors['bg'])
        
        # Get node positions
        pos = nx.get_node_attributes(self.graph, 'pos')
        if not pos and self.coordinates:
            pos = {node: (coord[1], coord[0]) for node, coord in self.coordinates.items()}
        
        num_nodes = len(self.graph.nodes())
        
        # CRITICAL: Force labels for Kansas Towns and KC Metro
        is_city_graph = "Kansas" in self.current_graph_type or "KC Metro" in self.current_graph_type
        
        # Adjust sizes based on graph type
        if is_city_graph:
            # For city graphs, ALWAYS show labels with readable size
            if num_nodes <= 50:
                node_size = 700
                font_size = 7
                edge_width = 1.5
            else:
                node_size = 400
                font_size = 6
                edge_width = 1.0
        else:
            # For generated graphs
            if num_nodes < 20:
                node_size = 800
                font_size = 8
                edge_width = 2
            elif num_nodes < 50:
                node_size = 500
                font_size = 7
                edge_width = 1.5
            elif num_nodes < 100:
                node_size = 300
                font_size = 6
                edge_width = 1
            else:
                node_size = 200
                font_size = 5
                edge_width = 0.8
        
        # Draw edges
        nx.draw_networkx_edges(self.graph, pos, ax=self.ax, 
                              edge_color=self.colors['edge'], 
                              alpha=0.4, width=edge_width)
        
        # Draw default nodes
        nx.draw_networkx_nodes(self.graph, pos, ax=self.ax, 
                              node_color=self.colors['default'], 
                              node_size=node_size, 
                              edgecolors=self.colors['text'],
                              linewidths=1.5, alpha=0.9)
        
        # CRITICAL: ALWAYS draw labels for city graphs
        if is_city_graph or num_nodes <= 50:
            labels = {}
            for node in self.graph.nodes():
                label = str(node)
                # Only truncate very long names
                if len(label) > 15:
                    label = label[:13] + ".."
                labels[node] = label
            
            nx.draw_networkx_labels(self.graph, pos, labels, ax=self.ax, 
                                   font_size=font_size, 
                                   font_color=self.colors['text'],
                                   font_weight='normal',
                                   font_family='sans-serif')
        
        # Highlight path
        if highlight_path and len(highlight_path) > 1:
            path_edges = [(highlight_path[i], highlight_path[i+1]) 
                         for i in range(len(highlight_path)-1)]
            nx.draw_networkx_edges(self.graph, pos, edgelist=path_edges,
                                 ax=self.ax, edge_color=self.colors['path'], 
                                 width=edge_width*2.5, alpha=0.9)
            nx.draw_networkx_nodes(self.graph, pos, nodelist=highlight_path,
                                 ax=self.ax, node_color=self.colors['path'], 
                                 node_size=node_size*1.2, edgecolors='white',
                                 linewidths=2.5)
            
            # Redraw path labels prominently
            if is_city_graph or num_nodes <= 50:
                path_labels = {node: labels.get(node, str(node)) for node in highlight_path}
                nx.draw_networkx_labels(self.graph, pos, path_labels, ax=self.ax,
                                       font_size=font_size+1,
                                       font_color='white',
                                       font_weight='bold')
        
        # Highlight start and goal
        start = self.start_var.get()
        goal = self.goal_var.get()
        
        if start and start in self.graph.nodes():
            nx.draw_networkx_nodes(self.graph, pos, nodelist=[start],
                                 ax=self.ax, node_color=self.colors['start'], 
                                 node_size=node_size*1.5, edgecolors='white',
                                 linewidths=3)
        
        if goal and goal in self.graph.nodes() and goal != start:
            nx.draw_networkx_nodes(self.graph, pos, nodelist=[goal],
                                 ax=self.ax, node_color=self.colors['goal'], 
                                 node_size=node_size*1.5, edgecolors='white',
                                 linewidths=3)
        
        # Title
        if highlight_path:
            title = f"🔍 {self.current_graph_type} - Path Found!"
        elif start and goal:
            title = f"🔍 {self.current_graph_type} - Ready to Search"
        else:
            title = f"🔍 {self.current_graph_type}"
        
        self.ax.set_title(title, fontsize=14, color=self.colors['text'], 
                         pad=15, fontweight='bold')
        self.ax.axis('off')
        self.fig.tight_layout()
        self.canvas.draw()
    
    def run_search(self):
        """Run the selected search algorithm"""
        if not self.graph:
            messagebox.showerror("Error", "❌ Please load or generate a graph first!")
            return
            
        start = self.start_var.get()
        goal = self.goal_var.get()
        
        if not start or not goal:
            messagebox.showerror("Error", "❌ Please select start and goal nodes!")
            return
        
        if start not in self.graph.nodes():
            messagebox.showerror("Error", f"❌ Start node '{start}' not found!")
            return
            
        if goal not in self.graph.nodes():
            messagebox.showerror("Error", f"❌ Goal node '{goal}' not found!")
            return
        
        if start == goal:
            messagebox.showerror("Error", "❌ Start and goal must be different!")
            return
        
        self.metrics_text.delete(1.0, tk.END)
        self.metrics_text.insert(tk.END, "⏳ Running search algorithm...\n\nPlease wait...")
        self.root.update()
        
        self.animation_frames = []
        self.current_frame = 0
        self.progress_var.set(0)
        self.progress_text.config(text="0/0")
        self.stop_animation()
        
        thread = threading.Thread(target=self._search_thread, args=(start, goal))
        thread.daemon = True
        thread.start()
    
    def _search_thread(self, start, goal):
        """Run search with animation capture"""
        try:
            algo_name = self.algo_var.get()
            heuristic = self.heuristic_var.get()
            
            if algo_name == "BFS":
                algorithm = BFS(self.graph, self.coordinates)
            elif algo_name == "DFS":
                algorithm = DFS(self.graph, self.coordinates)
            elif algo_name == "IDDFS":
                algorithm = IDDFS(self.graph, self.coordinates)
            elif algo_name == "BestFirst":
                algorithm = BestFirst(self.graph, self.coordinates, heuristic)
            elif algo_name == "AStar":
                algorithm = AStar(self.graph, self.coordinates, heuristic)
            else:
                self.root.after(0, lambda: messagebox.showerror("Error", "❌ Unknown algorithm!"))
                return
            
            self.animation_frames = []
            self.current_frame = 0
            
            self._setup_animation_capture(algorithm)
            path, metrics = algorithm.search(start, goal)
            
            self._capture_animation_state(algorithm, "complete", path if path else [])
            self.root.after(0, lambda: self._display_animation_results(path, metrics, algo_name))
            
        except Exception as e:
            import traceback
            error_msg = f"❌ Search failed: {str(e)}\n\n{traceback.format_exc()}"
            self.root.after(0, lambda: messagebox.showerror("Error", error_msg))
            self.root.after(0, lambda: self._reset_metrics_text())
    
    def _setup_animation_capture(self, algorithm):
        """Set up animation capture hooks"""
        def capture_state(state_type, current=None, frontier=None, explored=None, path=None):
            state = {
                'type': state_type,
                'current': current,
                'frontier': list(frontier) if frontier else [],
                'explored': set(explored) if explored else set(),
                'path': list(path) if path else []
            }
            self.animation_frames.append(state)
            if hasattr(self, 'root'):
                self.root.after(0, self._update_animation_progress)
        
        algorithm._original_search = getattr(algorithm, 'search', None)
        
        def animated_search(start, goal):
            capture_state("start", current=start, frontier=[start], explored=set())
            
            if isinstance(algorithm, BFS):
                queue = deque([start])
                setattr(algorithm, 'parent', {start: None})
                setattr(algorithm, 'visited', {start})
                
                while queue:
                    current = queue.popleft()
                    capture_state("visit", current=current, frontier=list(queue), explored=algorithm.visited)
                    
                    if current == goal:
                        path = []
                        if hasattr(algorithm, '_reconstruct_path'):
                            try:
                                path = algorithm._reconstruct_path(start, goal)
                            except:
                                pass
                        capture_state("complete", path=path)
                        return path, getattr(algorithm, 'metrics', {})
                    
                    for neighbor in algorithm.graph.neighbors(current):
                        if neighbor not in algorithm.visited:
                            algorithm.visited.add(neighbor)
                            algorithm.parent[neighbor] = current
                            queue.append(neighbor)
                            capture_state("expand", current=current, frontier=list(queue), explored=algorithm.visited)
                
                capture_state("complete", path=[])
                return [], getattr(algorithm, 'metrics', {})
            
            elif isinstance(algorithm, DFS):
                stack = [start]
                setattr(algorithm, 'parent', {start: None})
                setattr(algorithm, 'visited', {start})
                
                while stack:
                    current = stack.pop()
                    capture_state("visit", current=current, frontier=stack.copy(), explored=algorithm.visited)
                    
                    if current == goal:
                        path = []
                        if hasattr(algorithm, '_reconstruct_path'):
                            try:
                                path = algorithm._reconstruct_path(start, goal)
                            except:
                                pass
                        capture_state("complete", path=path)
                        return path, getattr(algorithm, 'metrics', {})
                    
                    for neighbor in algorithm.graph.neighbors(current):
                        if neighbor not in algorithm.visited:
                            algorithm.visited.add(neighbor)
                            algorithm.parent[neighbor] = current
                            stack.append(neighbor)
                            capture_state("expand", current=current, frontier=stack.copy(), explored=algorithm.visited)
                
                capture_state("complete", path=[])
                return [], getattr(algorithm, 'metrics', {})
            
            else:
                if algorithm._original_search:
                    capture_state("sample_start", current=start, frontier=[start], explored=set())
                    result = algorithm._original_search(start, goal)
                    try:
                        path = result[0] if result and len(result) > 0 else []
                    except:
                        path = []
                    capture_state("complete", path=path)
                    return result
                else:
                    capture_state("complete", path=[])
                    return [], {}
        
        algorithm.search = animated_search
    
    def _capture_animation_state(self, algorithm, state_type, path=None):
        """Capture algorithm state"""
        state = {
            'type': state_type,
            'current': getattr(algorithm, 'current_node', None),
            'frontier': getattr(algorithm, 'frontier_nodes', []),
            'explored': getattr(algorithm, 'visited', set()),
            'path': path if path else []
        }
        self.animation_frames.append(state)
        if hasattr(self, 'root'):
            self.root.after(0, self._update_animation_progress)
    
    def _update_animation_progress(self):
        """Update progress bar"""
        total = len(self.animation_frames)
        if total == 0:
            self.progress_var.set(0)
            self.progress_text.config(text="0/0")
            return
        progress = int((self.current_frame / total) * 100)
        self.progress_var.set(progress)
        self.progress_text.config(text=f"{self.current_frame}/{total}")
    
    def _display_animation_results(self, path, metrics, algo_name):
        """Display results and prepare animation"""
        bf = metrics.get('branching_factors', [])
        avg_bf = sum(bf) / len(bf) if bf else 0
        
        self.metrics_text.delete(1.0, tk.END)
        metrics_text = f"🔍 Algorithm: {algo_name}\n"
        metrics_text += f"{'='*50}\n"
        metrics_text += f"🎯 Start: {self.start_var.get()}\n"
        metrics_text += f"🎯 Goal: {self.goal_var.get()}\n"
        metrics_text += f"📊 Graph: {self.current_graph_type}\n"
        metrics_text += f"{'='*50}\n\n"
        
        if path:
            metrics_text += f"✅ Path Found: Yes ✓\n"
            metrics_text += f"📏 Path Length: {len(path)} nodes\n"
            metrics_text += f"💰 Path Cost: {metrics.get('path_cost', 0):.2f}\n"
            metrics_text += f"📊 Solution Depth: {metrics.get('solution_depth', 0)}\n\n"
        else:
            metrics_text += f"❌ Path Found: No ✗\n\n"
        
        metrics_text += f"{'='*50}\n"
        metrics_text += f"Performance Metrics:\n"
        metrics_text += f"{'='*50}\n"
        metrics_text += f"📊 Nodes Expanded: {metrics.get('nodes_expanded', 0)}\n"
        metrics_text += f"📊 Nodes Generated: {metrics.get('nodes_generated', 0)}\n"
        metrics_text += f"📊 Max Frontier Size: {metrics.get('max_frontier_size', 0)}\n"
        metrics_text += f"📊 Max Explored Size: {metrics.get('max_explored_size', 0)}\n"
        metrics_text += f"⏱️ Execution Time: {metrics.get('execution_time', 0)*1000:.2f} ms\n"
        metrics_text += f"🌳 Avg Branching Factor: {avg_bf:.2f}\n"
        
        if path and len(path) > 0:
            metrics_text += f"\n{'='*50}\n"
            metrics_text += f"🛤️ Path:\n"
            metrics_text += f"{'='*50}\n"
            if len(path) <= 15:
                metrics_text += " → ".join(path)
            else:
                metrics_text += " → ".join(path[:5])
                metrics_text += f"\n   ⋮ ({len(path)-10} intermediate)\n   ⋮\n"
                metrics_text += " → ".join(path[-5:])
        
        self.metrics_text.insert(1.0, metrics_text)
        self.draw_graph(highlight_path=path if path else None)
        
        if len(self.animation_frames) > 1:
            self.current_frame = 0
            self.show_frame(self.current_frame)
            messagebox.showinfo("Search Complete", 
                              f"✅ Animation ready!\n\n"
                              f"Use controls to step through {len(self.animation_frames)} frames.")
    
    def show_frame(self, frame_index):
        """Display animation frame"""
        if 0 <= frame_index < len(self.animation_frames):
            self.current_frame = frame_index
            frame = self.animation_frames[frame_index]
            self.draw_animated_graph(frame)
            self.update_frontier_display(frame)
            self._update_animation_progress()
    
    def draw_animated_graph(self, frame):
        """Draw graph with animation state"""
        if not self.graph:
            return
        
        self.ax.clear()
        self.ax.set_facecolor(self.colors['bg'])
        
        pos = nx.get_node_attributes(self.graph, 'pos')
        if not pos and self.coordinates:
            pos = {node: (coord[1], coord[0]) for node, coord in self.coordinates.items()}
        
        num_nodes = len(self.graph.nodes())
        is_city_graph = "Kansas" in self.current_graph_type or "KC Metro" in self.current_graph_type
        
        if is_city_graph:
            node_size = 700 if num_nodes <= 50 else 400
            font_size = 7 if num_nodes <= 50 else 6
            edge_width = 1.5 if num_nodes <= 50 else 1.0
        else:
            node_size, font_size, edge_width = self._get_graph_sizes(num_nodes)
        
        nx.draw_networkx_edges(self.graph, pos, ax=self.ax, 
                              edge_color=self.colors['edge'], alpha=0.4, width=edge_width)
        
        # Draw nodes by state
        all_nodes = set(self.graph.nodes())
        explored = set(frame.get('explored', set()))
        frontier = set(frame.get('frontier', []))
        current = frame.get('current')
        path_nodes = set(frame.get('path', []))
        
        # Unvisited nodes
        unvisited = all_nodes - explored - frontier - path_nodes
        if current:
            unvisited.discard(current)
        
        if unvisited:
            nx.draw_networkx_nodes(self.graph, pos, nodelist=list(unvisited),
                                  ax=self.ax, node_color=self.colors['default'],
                                  node_size=node_size, alpha=0.6)
        
        # Explored nodes
        if explored:
            nx.draw_networkx_nodes(self.graph, pos, nodelist=list(explored),
                                  ax=self.ax, node_color=self.colors['explored'],
                                  node_size=node_size, alpha=0.7)
        
        # Frontier nodes
        if frontier:
            nx.draw_networkx_nodes(self.graph, pos, nodelist=list(frontier),
                                  ax=self.ax, node_color=self.colors['frontier'],
                                  node_size=node_size, alpha=0.9)
        
        # Current node
        if current:
            nx.draw_networkx_nodes(self.graph, pos, nodelist=[current],
                                  ax=self.ax, node_color=self.colors['current'],
                                  node_size=node_size * 1.5, alpha=1.0)
        
        # Path
        if frame.get('path') and len(frame['path']) > 1:
            path = frame['path']
            path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
            nx.draw_networkx_edges(self.graph, pos, edgelist=path_edges,
                                 ax=self.ax, edge_color=self.colors['path'],
                                 width=edge_width * 3, alpha=0.9)
        
        # Start and goal
        start = self.start_var.get()
        goal = self.goal_var.get()
        
        if start in self.graph.nodes():
            nx.draw_networkx_nodes(self.graph, pos, nodelist=[start],
                                 ax=self.ax, node_color=self.colors['start'],
                                 node_size=node_size * 1.8, alpha=1.0)
        
        if goal in self.graph.nodes() and goal != start:
            nx.draw_networkx_nodes(self.graph, pos, nodelist=[goal],
                                 ax=self.ax, node_color=self.colors['goal'],
                                 node_size=node_size * 1.8, alpha=1.0)
        
        # CRITICAL: Always show labels for city graphs
        if is_city_graph or num_nodes <= 50:
            labels = {}
            for node in self.graph.nodes():
                label = str(node)
                if len(label) > 15:
                    label = label[:13] + ".."
                labels[node] = label
            nx.draw_networkx_labels(self.graph, pos, labels, ax=self.ax,
                                   font_size=font_size, font_color=self.colors['text'])
        
        frame_type = frame.get('type', 'searching').title()
        title = f"🔍 {self.current_graph_type} - {frame_type} (Frame {self.current_frame + 1}/{len(self.animation_frames)})"
        self.ax.set_title(title, fontsize=12, color=self.colors['text'], pad=15)
        self.ax.axis('off')
        self.fig.tight_layout()
        self.canvas.draw()
    
    def _get_graph_sizes(self, num_nodes):
        """Get appropriate sizes"""
        if num_nodes < 20:
            return 800, 8, 2
        elif num_nodes < 50:
            return 500, 7, 1.5
        elif num_nodes < 100:
            return 300, 6, 1
        else:
            return 200, 5, 0.8
    
    def update_frontier_display(self, frame):
        """Update frontier display"""
        text = "🎯 Current Search State\n" + "=" * 30 + "\n\n"
        
        if frame.get('current'):
            text += f"🔹 Current Node: {frame['current']}\n\n"
        
        if frame.get('frontier'):
            frontier_nodes = frame['frontier']
            text += f"🟡 Frontier Nodes ({len(frontier_nodes)}):\n"
            for i, node in enumerate(frontier_nodes[:10]):
                text += f"  {i+1}. {node}\n"
            if len(frontier_nodes) > 10:
                text += f"  ... and {len(frontier_nodes) - 10} more\n"
        else:
            text += "🟡 Frontier: Empty\n"
        
        if frame.get('explored'):
            text += f"\n🔵 Explored: {len(frame['explored'])} nodes\n"
        
        if frame.get('type') == 'complete':
            if frame.get('path'):
                text += f"\n💗 Path Found! Length: {len(frame['path'])}\n"
            else:
                text += f"\n❌ No Path Found\n"
        
        self.frontier_text.delete(1.0, tk.END)
        self.frontier_text.insert(tk.END, text)
    
    # Animation controls
    def first_frame(self):
        if self.animation_frames:
            self.stop_animation()
            self.current_frame = 0
            self.show_frame(self.current_frame)

    def prev_frame(self):
        if self.animation_frames and self.current_frame > 0:
            self.stop_animation()
            self.current_frame -= 1
            self.show_frame(self.current_frame)

    def next_frame(self):
        if self.animation_frames and self.current_frame < len(self.animation_frames) - 1:
            self.stop_animation()
            self.current_frame += 1
            self.show_frame(self.current_frame)

    def last_frame(self):
        if self.animation_frames:
            self.stop_animation()
            self.current_frame = len(self.animation_frames) - 1
            self.show_frame(self.current_frame)

    def toggle_play(self):
        if self.is_animating:
            self.stop_animation()
        else:
            self.start_animation()

    def start_animation(self):
        if not self.animation_frames:
            return
        self.is_animating = True
        try:
            self.play_button.config(text="⏸️")
        except:
            pass
        
        def animate():
            if self.is_animating and self.current_frame < len(self.animation_frames) - 1:
                self.current_frame += 1
                self.show_frame(self.current_frame)
                self.animation_job = self.root.after(self.animation_delay, animate)
            else:
                self.stop_animation()
        
        self.animation_job = self.root.after(self.animation_delay, animate)

    def stop_animation(self):
        self.is_animating = False
        try:
            self.play_button.config(text="▶️")
        except:
            pass
        if self.animation_job:
            try:
                self.root.after_cancel(self.animation_job)
            except:
                pass
            self.animation_job = None

    def update_animation_speed(self, event=None):
        speed = self.speed_var.get() if hasattr(self, 'speed_var') else 1.0
        if speed <= 0:
            speed = 0.1
        self.animation_delay = int(500 / speed)
        try:
            self.speed_label.config(text=f"{speed:.1f}x")
        except:
            pass
        if self.is_animating:
            self.stop_animation()
            self.start_animation()
    
    def reset_visualization(self):
        """Reset all visualization state"""
        self.stop_animation()
        self.animation_frames = []
        self.current_frame = 0
        self.progress_var.set(0)
        self.progress_text.config(text="0/0")
        self.draw_graph()
        self.frontier_text.delete(1.0, tk.END)
        self.frontier_text.insert(tk.END, "Run a search to see frontier nodes...")
        self._reset_metrics_text()
    
    def _reset_metrics_text(self):
        """Reset metrics text to default"""
        self.metrics_text.delete(1.0, tk.END)
        self.metrics_text.insert(tk.END, "🔍 Search metrics will appear here after running an algorithm.\n\n")
        self.metrics_text.insert(tk.END, "📋 Available metrics:\n")
        self.metrics_text.insert(tk.END, "  • Path found (Y/N)\n")
        self.metrics_text.insert(tk.END, "  • Path length and cost\n")
        self.metrics_text.insert(tk.END, "  • Nodes expanded/generated\n")
        self.metrics_text.insert(tk.END, "  • Execution time\n")
        self.metrics_text.insert(tk.END, "  • Memory usage\n")
        self.metrics_text.insert(tk.END, "  • Branching factors\n")
        self.metrics_text.insert(tk.END, "  • Frontier size\n")
    
    # Benchmarking methods
    def run_single_benchmark(self):
        """Run single benchmark"""
        if not self.graph:
            messagebox.showerror("Error", "❌ Please load or generate a graph first!")
            return
        
        start = self.start_var.get()
        goal = self.goal_var.get()
        algo_name = self.algo_var.get()
        heuristic = self.heuristic_var.get()
        
        if not start or not goal:
            messagebox.showerror("Error", "❌ Please select start and goal nodes!")
            return
        
        # Initialize algorithm
        if algo_name == "BFS":
            algorithm = lambda s, g: BFS(self.graph, self.coordinates).search(s, g)
        elif algo_name == "DFS":
            algorithm = lambda s, g: DFS(self.graph, self.coordinates).search(s, g)
        elif algo_name == "IDDFS":
            algorithm = lambda s, g: IDDFS(self.graph, self.coordinates).search(s, g)
        elif algo_name == "BestFirst":
            algorithm = lambda s, g: BestFirst(self.graph, self.coordinates, heuristic).search(s, g)
        elif algo_name == "AStar":
            algorithm = lambda s, g: AStar(self.graph, self.coordinates, heuristic).search(s, g)
        else:
            messagebox.showerror("Error", "❌ Unknown algorithm!")
            return
        
        # Run benchmark
        results = self.benchmark_harness.single_run(
            algorithm, algo_name, start, goal, self.current_graph_type
        )
        
        # Display results
        self.benchmark_text.delete(1.0, tk.END)
        self.benchmark_text.insert(tk.END, f"📊 Benchmark Results for {algo_name}\n")
        self.benchmark_text.insert(tk.END, f"{'='*50}\n\n")
        self.benchmark_text.insert(tk.END, f"Graph: {self.current_graph_type}\n")
        self.benchmark_text.insert(tk.END, f"Start: {start}\n")
        self.benchmark_text.insert(tk.END, f"Goal: {goal}\n\n")
        
        for key, value in results.items():
            if key in ['algorithm', 'graph', 'start', 'goal']:
                continue
            if isinstance(value, float):
                self.benchmark_text.insert(tk.END, f"• {key}: {value:.4f}\n")
            else:
                self.benchmark_text.insert(tk.END, f"• {key}: {value}\n")
        
        messagebox.showinfo("Benchmark Complete", "✅ Benchmark completed successfully!")
    
    def run_batch_benchmark(self):
        """Run batch benchmark comparing all algorithms"""
        if not self.graph:
            messagebox.showerror("Error", "❌ Please load or generate a graph first!")
            return
        
        start = self.start_var.get()
        goal = self.goal_var.get()
        heuristic = self.heuristic_var.get()
        
        if not start or not goal:
            messagebox.showerror("Error", "❌ Please select start and goal nodes!")
            return
        
        # Show progress
        self.benchmark_text.delete(1.0, tk.END)
        self.benchmark_text.insert(tk.END, "⏳ Running batch benchmark...\n\nThis may take a moment...")
        self.root.update()
        
        # Define all algorithms
        algorithms = {
            "BFS": lambda s, g: BFS(self.graph, self.coordinates).search(s, g),
            "DFS": lambda s, g: DFS(self.graph, self.coordinates).search(s, g),
            "IDDFS": lambda s, g: IDDFS(self.graph, self.coordinates).search(s, g),
            "BestFirst": lambda s, g: BestFirst(self.graph, self.coordinates, heuristic).search(s, g),
            "AStar": lambda s, g: AStar(self.graph, self.coordinates, heuristic).search(s, g)
        }
        
        test_cases = [{'start': start, 'goal': goal, 'graph': self.current_graph_type, 'name': 'Test 1', 'seed': 42}]
        
        # Run batch benchmark
        results = self.benchmark_harness.batch_run(algorithms, test_cases, num_repeats=5)
        
        # Generate statistics
        stats = self.benchmark_harness.generate_statistics(results)
        
        # Display results
        self.benchmark_text.delete(1.0, tk.END)
        self.benchmark_text.insert(tk.END, "📊 Batch Benchmark Results\n")
        self.benchmark_text.insert(tk.END, f"{'='*50}\n\n")
        self.benchmark_text.insert(tk.END, f"Graph: {self.current_graph_type}\n")
        self.benchmark_text.insert(tk.END, f"Start: {start} → Goal: {goal}\n")
        self.benchmark_text.insert(tk.END, f"Runs per algorithm: 5\n\n")
        
        for algo_name, stat in stats.items():
            self.benchmark_text.insert(tk.END, f"🔹 {algo_name}:\n")
            self.benchmark_text.insert(tk.END, f"   Success Rate: {stat['success_rate']*100:.1f}%\n")
            self.benchmark_text.insert(tk.END, f"   Avg Time: {stat['avg_time']*1000:.2f} ± {stat['std_time']*1000:.2f} ms\n")
            self.benchmark_text.insert(tk.END, f"   Avg Nodes Expanded: {stat['avg_nodes_expanded']:.0f}\n")
            self.benchmark_text.insert(tk.END, f"   Avg Path Cost: {stat['avg_path_cost']:.2f}\n")
            self.benchmark_text.insert(tk.END, f"   Avg Memory: {stat['avg_memory']:.2f} KB\n\n")
        
        messagebox.showinfo("Benchmark Complete", "✅ Batch benchmark completed!\n\nResults show mean ± std dev across 5 runs.")
    
    def export_results(self):
        """Export benchmark results to CSV"""
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
                title="Export Benchmark Results"
            )
            if filename:
                start = self.start_var.get()
                goal = self.goal_var.get()
                heuristic = self.heuristic_var.get()
                
                if not start or not goal:
                    messagebox.showerror("Error", "❌ Please select start and goal nodes first!")
                    return
                
                algorithms = {
                    "BFS": lambda s, g: BFS(self.graph, self.coordinates).search(s, g),
                    "DFS": lambda s, g: DFS(self.graph, self.coordinates).search(s, g),
                    "IDDFS": lambda s, g: IDDFS(self.graph, self.coordinates).search(s, g),
                    "BestFirst": lambda s, g: BestFirst(self.graph, self.coordinates, heuristic).search(s, g),
                    "AStar": lambda s, g: AStar(self.graph, self.coordinates, heuristic).search(s, g)
                }
                
                test_cases = [{'start': start, 'goal': goal, 'graph': self.current_graph_type, 'name': 'Export', 'seed': 42}]
                results = self.benchmark_harness.batch_run(algorithms, test_cases, num_repeats=5)
                
                self.benchmark_harness.export_to_csv(results, filename)
                messagebox.showinfo("Success", f"✅ Results exported to:\n{filename}")
        except Exception as e:
            messagebox.showerror("Error", f"❌ Failed to export: {str(e)}")

def main():
    """Main function to run the GUI"""
    root = tk.Tk()
    app = SearchVisualizationGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()