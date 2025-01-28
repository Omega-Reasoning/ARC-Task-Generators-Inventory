import random
import numpy as np

from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from input_library import create_object, retry
from transformation_library import find_connected_objects

class TaskKyDurxM8poCvnuRTrzFV2RGenerator(ARCTaskGenerator):
    def __init__(self):
        # Define the input reasoning chain
        input_reasoning_chain = [
            "All input grids are squares.",
            "Let n be the height and width of an input grid, with n being an odd number.",
            "Each input grid contains the four corner cells colored, while all other cells are empty (0).",
            "The four corner cells are colored with {color('cell_color1')}, {color('cell_color2')}, {color('cell_color3')}, and {color('cell_color4')}."
        ]

        # Define the transformation reasoning chain
        transformation_reasoning_chain = [
            "The output grid is created by copying the input grid and filling all empty cells with {color('fill_color')}, except for the four cells which are adjacent (up, down, left, right) to the middle cell of the grid, which remain empty (0)."
        ]
        
        # Initialize the superclass
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        # Choose an odd dimension n in [5..30] for variety
        n = random.choice([d for d in range(5, 30) if d % 2 == 1])
        
        # Initialize an empty grid
        grid = np.zeros((n, n), dtype=int)
        
        # Shuffle the corners for variation
        corners = [(0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1)]
        random.shuffle(corners)

        # Assign colors to the corners
        grid[corners[0]] = taskvars['cell_color1']
        grid[corners[1]] = taskvars['cell_color2']
        grid[corners[2]] = taskvars['cell_color3']
        grid[corners[3]] = taskvars['cell_color4']
        
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        fill_color = taskvars['fill_color']
        n = grid.shape[0]
        
        # Copy input grid
        output = grid.copy()
        
        # Identify the center cell
        center_r, center_c = (n // 2, n // 2)
        
        # Define the four adjacent positions
        adjacent_positions = [(center_r - 1, center_c), (center_r + 1, center_c),
                              (center_r, center_c - 1), (center_r, center_c + 1)]
        
        # Fill all empty cells with fill_color except the adjacent positions
        for r in range(n):
            for c in range(n):
                if output[r, c] == 0 and (r, c) not in adjacent_positions:
                    output[r, c] = fill_color
        
        return output
    
    def create_grids(self):
        # Pick 5 distinct colors for cell_color1, cell_color2, cell_color3, cell_color4, and fill_color
        colors = random.sample(range(1, 10), 5)
        taskvars = {
            'cell_color1': colors[0],
            'cell_color2': colors[1],
            'cell_color3': colors[2],
            'cell_color4': colors[3],
            'fill_color':  colors[4]
        }
        
        # Decide the number of training examples (3 or 4)
        nr_train = random.choice([3, 4])
        
        train = []
        for _ in range(nr_train):
            inp = self.create_input(taskvars, {})
            out = self.transform_input(inp, taskvars)
            train.append({'input': inp, 'output': out})
        
        # Create one test example
        test_inp = self.create_input(taskvars, {})
        test_out = self.transform_input(test_inp, taskvars)
        test = [{'input': test_inp, 'output': test_out}]
        
        return taskvars, {'train': train, 'test': test}

