from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from Framework.input_library import retry

class Task6150a2bdGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['grid_size']} x {vars['grid_size']}.",
            "They contain multi-colored (1–9) cells and empty (0) cells.",
            "The grid is conceptually divided into three parts: the inverse diagonal (from top-right to bottom-left), the area above the inverse diagonal, and the area below it.",
            "The area below the inverse diagonal is completely empty (0).",
            "The area above the inverse diagonal is completely filled with multi-colored cells.",
            "The inverse diagonal is mostly filled with multi-colored cells but may occasionally contain some empty (0) cells.",
            "There should always be an object made of at least two, 4-way connected, same-colored cells that touches the top-left corner of each input grid."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid and reflecting it along the inverse diagonal (from top-right to bottom-left).",
            "All grid cells above the inverse diagonal are reflected below it, and all cells below the diagonal are reflected above it.",
            "This results in empty (0) cells that were originally below the inverse diagonal appearing above it.",
            "The inverse diagonal remains unchanged in the output grid."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self) -> tuple[dict[str, any], TrainTestData]:
        
        grid_size = random.randint(5, 30)
        taskvars = {'grid_size': grid_size}
        
        # Generate train and test grids
        train_grids = []
        
        # Create 3 grids with fully filled inverse diagonal
        for _ in range(3):
            input_grid = self.create_input(taskvars, {'empty_diagonal': False})
            output_grid = self.transform_input(input_grid, taskvars)
            train_grids.append({'input': input_grid, 'output': output_grid})
        
        # Create 1 grid with some empty cells in the inverse diagonal
        input_grid = self.create_input(taskvars, {'empty_diagonal': True})
        output_grid = self.transform_input(input_grid, taskvars)
        train_grids.append({'input': input_grid, 'output': output_grid})
        
        # Create test grid
        test_input = self.create_input(taskvars, {'empty_diagonal': random.choice([True, False])})
        test_output = self.transform_input(test_input, taskvars)
        
        return taskvars, {
            'train': train_grids,
            'test': [{'input': test_input, 'output': test_output}]
        }
    
    def create_input(self, taskvars, gridvars):
        grid_size = taskvars['grid_size']
        empty_diagonal = gridvars.get('empty_diagonal', False)
        
        # Initialize grid with zeros
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Fill area above inverse diagonal with random colors (1-9)
        for r in range(grid_size):
            for c in range(grid_size):
                # Calculate if cell is on or above the inverse diagonal
                # Inverse diagonal: r + c == grid_size - 1
                if r + c <= grid_size - 1:
                    # On inverse diagonal
                    if r + c == grid_size - 1:
                        if not empty_diagonal or random.random() > 0.3:  # 70% chance to fill if empty_diagonal=True
                            grid[r, c] = random.randint(1, 9)
                    # Above inverse diagonal
                    else:
                        grid[r, c] = random.randint(1, 9)
        
        # Create a 4-way connected object at the top-left corner
        color = random.randint(1, 9)
        grid[0, 0] = color
        
        # Extend the object by creating at least one more connected cell
        if grid_size > 1:
            # Add at least one connected cell
            if random.choice([True, False]):
                grid[0, 1] = color
            else:
                grid[1, 0] = color
        
        return grid
    
    def transform_input(self, grid, taskvars):
        grid_size = taskvars['grid_size']
        output = grid.copy()
        
        # Reflect cells across the inverse diagonal
        for r in range(grid_size):
            for c in range(grid_size):
                # If it's not on the inverse diagonal (r + c != grid_size - 1)
                if r + c != grid_size - 1:
                    # Reflection coordinates across inverse diagonal
                    reflect_r = grid_size - 1 - c
                    reflect_c = grid_size - 1 - r
                    output[reflect_r, reflect_c] = grid[r, c]
        
        return output