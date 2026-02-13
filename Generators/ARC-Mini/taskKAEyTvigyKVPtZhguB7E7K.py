from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from typing import Dict, Any, Tuple, List
from input_library import retry

class TaskKAEyTvigyKVPtZhguB7E7KGenerator(ARCTaskGenerator):
    
    def __init__(self):
        input_reasoning_chain = [
            "The input grids are of different sizes.",
            "Each input grid contains exactly two completely filled columns of {color('col_color')} color.",
            "The two colored columns must not be consecutive.",
            "All other cells are empty (0)."
        ]
        
        transformation_reasoning_chain = [
            "The output grids are constructed by copying the input grids and identifying the two {color('col_color')} columns.",
            "All columns in between these two {color('col_color')} columns are filled with {color('fill_color')} color."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        height = gridvars['height']
        width = gridvars['width']
        col_color = taskvars['col_color']
        col1_pos = gridvars['col1_pos']
        col2_pos = gridvars['col2_pos']
        
        # Create empty grid
        grid = np.zeros((height, width), dtype=int)
        
        # Fill the two columns completely
        grid[:, col1_pos] = col_color
        grid[:, col2_pos] = col_color
        
        return grid

    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        output_grid = grid.copy()
        col_color = taskvars['col_color']
        fill_color = taskvars['fill_color']
        
        # Find the two colored columns
        colored_cols = []
        for col in range(grid.shape[1]):
            if np.all(grid[:, col] == col_color) and col_color != 0:
                colored_cols.append(col)
        
        if len(colored_cols) == 2:
            left_col = min(colored_cols)
            right_col = max(colored_cols)
            
            # Fill all columns between the two colored columns
            for col in range(left_col + 1, right_col):
                output_grid[:, col] = fill_color
        
        return output_grid

    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Generate task variables
        col_color = random.randint(1, 9)
        fill_color = random.randint(1, 9)
        # Ensure fill_color is different from col_color
        while fill_color == col_color:
            fill_color = random.randint(1, 9)
        
        taskvars = {
            'col_color': col_color,
            'fill_color': fill_color
        }
        
        # Generate 3-5 train examples and 1 test example
        num_train = random.randint(3, 5)
        used_sizes = set()
        
        def generate_example():
            def try_generate():
                # Generate grid size (between 5 and 30)
                height = random.randint(5, 30)
                width = random.randint(5, 30)
                size = (height, width)
                
                # Ensure all grids have different sizes
                if size in used_sizes:
                    return None
                
                # Generate two non-consecutive column positions
                if width < 3:  # Need at least 3 columns for non-consecutive constraint
                    return None
                
                col1_pos = random.randint(0, width - 1)
                col2_pos = random.randint(0, width - 1)
                
                # Ensure columns are not consecutive and not the same
                if abs(col1_pos - col2_pos) <= 1:
                    return None
                
                used_sizes.add(size)
                return {
                    'height': height,
                    'width': width,
                    'col1_pos': col1_pos,
                    'col2_pos': col2_pos
                }
            
            return retry(try_generate, lambda x: x is not None, max_attempts=100)
        
        # Generate training examples
        train_examples = []
        for _ in range(num_train):
            gridvars = generate_example()
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            train_examples.append({'input': input_grid, 'output': output_grid})
        
        # Generate test example
        gridvars = generate_example()
        input_grid = self.create_input(taskvars, gridvars)
        output_grid = self.transform_input(input_grid, taskvars)
        test_examples = [{'input': input_grid, 'output': output_grid}]
        
        train_test_data = {
            'train': train_examples,
            'test': test_examples
        }
        
        return taskvars, train_test_data


