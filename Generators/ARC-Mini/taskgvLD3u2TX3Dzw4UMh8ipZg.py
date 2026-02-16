from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random

class TaskgvLD3u2TX3Dzw4UMh8ipZgGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['grid_size']}x{vars['grid_size']}.",
            "They have the main diagonal (top-left to bottom-right) and cell at position (0,{vars['grid_size']}) filled, with multi-colored cells.",
            "The remaining cells are empty (0)."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid and swapping the cells at positions (0,0) and (0,{vars['grid_size']}) and removing the cell at position ({vars['grid_size']},{vars['grid_size']})."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        grid_size = taskvars['grid_size']
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Randomly assign colors to the diagonal and extra cell
        diagonal_colors = [random.randint(1, 9) for _ in range(grid_size)]
        extra_cell_color = random.randint(1, 9)
        
        for i in range(grid_size):
            grid[i, i] = diagonal_colors[i]
        grid[0, grid_size - 1] = extra_cell_color
        
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        grid_size = taskvars['grid_size']
        output_grid = grid.copy()
        
        # Swap (0,0) and (0, grid_size-1)
        output_grid[0, 0], output_grid[0, grid_size - 1] = output_grid[0, grid_size - 1], output_grid[0, 0]
        
        # Remove the cell at (grid_size-1, grid_size-1)
        output_grid[grid_size - 1, grid_size - 1] = 0
        
        return output_grid
    
    def create_grids(self) -> tuple:
        grid_size = random.randint(5, 30)
        taskvars = {'grid_size': grid_size}
        
        num_train_examples = random.randint(3, 6)
        train_data = []
        
        for _ in range(num_train_examples):
            input_grid = self.create_input(taskvars, {})
            output_grid = self.transform_input(input_grid, taskvars)
            train_data.append({'input': input_grid, 'output': output_grid})
        
        # Generate test case
        test_input_grid = self.create_input(taskvars, {})
        test_output_grid = self.transform_input(test_input_grid, taskvars)
        test_data = [{'input': test_input_grid, 'output': test_output_grid}]
        
        return taskvars, {'train': train_data, 'test': test_data}

