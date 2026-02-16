from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from Framework.input_library import create_object, random_cell_coloring
from Framework.transformation_library import find_connected_objects

class Task3bd67248Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are square and can have different sizes.",
            "They contain a completely filled first column with same-colored cells, while all other cells remain empty (0).",
            "The color of the first column varies across examples."
        ]
        
        transformation_reasoning_chain = [
            "The output grids are constructed by copying the input grids and coloring the entire last row, except for the bottom-left corner cell, with {color('fill_color1')} color.",
            "Next, color the entire inverse diagonal (top-right to bottom-left), except for the bottom-left corner cell, with {color('fill_color2')} color."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self) -> tuple[dict[str, any], TrainTestData]:
        # Define task variables
        fill_color1 = random.randint(1, 9)
        
        # Ensure fill_color2 is different from fill_color1
        available_colors = [c for c in range(1, 10) if c != fill_color1]
        fill_color2 = random.choice(available_colors)

        # Ensure column color is different from both fill_color1 and fill_color2
        available_column_colors = [c for c in range(1, 10) if c != fill_color1 and c != fill_color2]
        
        taskvars = {
            'fill_color1': fill_color1,
            'fill_color2': fill_color2
        }
        
        # Create train and test examples
        num_train_examples = random.randint(3, 5)
        
        # Ensure we use different sizes and column colors across examples
        used_sizes = set()
        used_column_colors = set()
        
        train_examples = []
        
        for _ in range(num_train_examples):
            # Make sure we don't repeat size
            size = random.randint(5, 15)
            while size in used_sizes:
                size = random.randint(5, 15)
            used_sizes.add(size)
            
            # Ensure column_color is different from fill_color1 and fill_color2
            column_color = random.choice(available_column_colors)
            while column_color in used_column_colors:
                column_color = random.choice(available_column_colors)
            used_column_colors.add(column_color)
            
            gridvars = {'size': size, 'column_color': column_color}
            
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Create test example
        test_size = random.randint(5, 20)
        while test_size in used_sizes:
            test_size = random.randint(5, 20)
            
        test_column_color = random.choice(available_column_colors)
        while test_column_color in used_column_colors:
            test_column_color = random.choice(available_column_colors)
            
        test_gridvars = {'size': test_size, 'column_color': test_column_color}
        
        test_input_grid = self.create_input(taskvars, test_gridvars)
        test_output_grid = self.transform_input(test_input_grid, taskvars)
        
        test_examples = [{
            'input': test_input_grid,
            'output': test_output_grid
        }]
        
        return taskvars, {
            'train': train_examples,
            'test': test_examples
        }
    
    def create_input(self, taskvars: dict[str, any], gridvars: dict[str, any]) -> np.ndarray:
        size = gridvars['size']
        column_color = gridvars['column_color']
        
        # Create an empty square grid
        grid = np.zeros((size, size), dtype=int)
        
        # Fill the first column with the specified color
        grid[:, 0] = column_color
        
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: dict[str, any]) -> np.ndarray:
        # Copy the input grid
        output_grid = grid.copy()
        
        size = grid.shape[0]  # grid is square, so width = height
        
        # Fill the last row, except bottom-left corner
        output_grid[size-1, 1:] = taskvars['fill_color1']
        
        # Fill the inverse diagonal (top-right to bottom-left), including top-right corner
        for i in range(size):  # Start from 0 to include the top-right corner
            row = i
            col = size - 1 - i
            
            # Skip bottom-left corner
            if row == size-1 and col == 0:
                continue
                
            output_grid[row, col] = taskvars['fill_color2']
        
        return output_grid

