# DiagonalFillTaskGenerator.py

from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from input_library import create_object
from transformation_library import find_connected_objects
import numpy as np
from typing import Dict, Any, Tuple
import random

class TasktaskAiZod7C33PQp4imyWrJuhcGenerator(ARCTaskGenerator):
    
    
    def __init__(self):
        # Initialize the input reasoning chain
        input_reasoning_chain = [
            "Input grids are of size {vars['grid_size']} x {vars['grid_size']}.",
            "Each input grid contains {vars['num_diag_cells']} same-colored cells, which are equally spaced along the main diagonal.",
            "The cells on the main diagonal can only be blue (1), green (3), or pink (6). All other cells are empty (0)."
        ]
        
        # Initialize the transformation reasoning chain
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid.",
            "For each colored cell; all adjacent empty (0) cells (up, down, left, right) are filled.",
            "The fill color is determined by the diagonal cell color:",
            "If the diagonal cells are blue (1), adjacent empty cells are filled with red (2).",
            "If the diagonal cells are green (3), adjacent empty cells are filled with yellow (4).",
            "If the diagonal cells are pink (6), adjacent empty cells are filled with orange (7)."
        ]
        
        # Initialize the superclass with the reasoning chains
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        grid_size = taskvars['grid_size']
        num_diag_cells = taskvars['num_diag_cells']
        color = gridvars['color']
        
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        step = (grid_size - 1) // (num_diag_cells - 1) if num_diag_cells > 1 else 0
        diag_positions = [i * step for i in range(num_diag_cells)]
        
        for pos in diag_positions:
            grid[pos, pos] = color
        
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        grid_size = taskvars['grid_size']
        color = gridvars['color']
        
        output_grid = np.copy(grid)
        
        fill_color_map = {
            1: 2,  # Blue (1) -> Red (2)
            3: 4,  # Green (3) -> Yellow (4)
            6: 7   # Pink (6) -> Orange (7)
        }
        fill_color = fill_color_map.get(color, 0)
        
        if fill_color == 0:
            return output_grid
        
        step = (grid_size - 1) // (taskvars['num_diag_cells'] - 1) if taskvars['num_diag_cells'] > 1 else 0
        diag_positions = [i * step for i in range(taskvars['num_diag_cells'])]
        
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        for pos in diag_positions:
            r, c = pos, pos
            for dr, dc in directions:
                nr, nc = r + dr, c + dc
                if 0 <= nr < grid_size and 0 <= nc < grid_size:
                    if output_grid[nr, nc] == 0:
                        output_grid[nr, nc] = fill_color
        
        return output_grid
    
    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        available_colors = [1, 3, 6]
        
        grid_size = random.choice([size for size in range(5, 31, 2)])
        num_diag_cells = (grid_size + 1) // 2
        
        taskvars = {
            'grid_size': grid_size,
            'num_diag_cells': num_diag_cells
        }
        
        random.shuffle(available_colors)
        train_colors = available_colors[:2]
        test_color = available_colors[2]
        
        train = []
        for color in train_colors:
            gridvars = {'color': color}
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars, gridvars)
            train.append({'input': input_grid, 'output': output_grid})
        
        test = []
        gridvars = {'color': test_color}
        input_grid = self.create_input(taskvars, gridvars)
        output_grid = self.transform_input(input_grid, taskvars, gridvars)
        test.append({'input': input_grid, 'output': output_grid})
        
        train_test_data = {
            'train': train,
            'test': test
        }
        
        return taskvars, train_test_data
