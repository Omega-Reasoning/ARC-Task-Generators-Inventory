from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.transformation_library import find_connected_objects
from Framework.input_library import retry
import numpy as np
import random
from typing import Dict, Any, Tuple, List

class Taskbd4472b8Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size (2*m+2) × m, where m is an integer at most equal to 8.",
            "In each input grid, the top row is filled with m different colors.",
            "The second row is filled with {color('color_1')} which is not one of those m colors.",
            "The remaining rows are all empty (0).",
            "The value of m, as well as the colors used to fill the first row, vary in each input grid to maintain diversity."
        ]
        
        transformation_reasoning_chain = [
            "The output is constructed by first copying the input grid.",
            "The remaining empty rows are then filled sequentially, each row being assigned a uniform color.",
            "The assignment of colors proceeds according to the left-to-right order of the first row.",
            "After the last color in the sequence has been applied, the assignment process returns to the first color and continues in the same cyclic manner.",
            "This cycle of uniform row assignments is repeated until all rows of the grid are completely filled."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars: Dict[str, Any], gridvars: Dict[str, Any]) -> np.ndarray:
        m = gridvars['m']
        height = 2 * m + 2
        width = m
        
        # Create empty grid
        grid = np.zeros((height, width), dtype=int)
        
        # Fill top row with m different colors
        top_row_colors = gridvars['top_row_colors']
        for col in range(m):
            grid[0, col] = top_row_colors[col]
        
        # Fill second row with uniform color (color_1)
        grid[1, :] = taskvars['color_1']
        
        # Remaining rows stay empty (0)
        return grid
    
    def transform_input(self, grid: np.ndarray, taskvars: Dict[str, Any]) -> np.ndarray:
        height, width = grid.shape
        m = width  # Since width = m
        
        # Start with a copy of the input
        output = grid.copy()
        
        # Get the colors from the first row for cycling
        first_row_colors = [grid[0, col] for col in range(m)]
        
        # Fill remaining empty rows (starting from row 2)
        color_index = 0
        for row in range(2, height):
            # Fill entire row with current color from cycle
            output[row, :] = first_row_colors[color_index]
            # Move to next color in cycle
            color_index = (color_index + 1) % m
        
        return output
    
    def create_grids(self) -> Tuple[Dict[str, Any], TrainTestData]:
        # Generate random number of training examples
        num_train = random.randint(3, 6)
        num_test = 1
        
        def create_single_example():
            # Choose m (varies per grid)
            m = random.randint(2, 8)  # m should be at least 2 to have meaningful cycles
            
            # Ensure grid size constraints (max 30x30)
            height = 2 * m + 2
            width = m
            if height > 30 or width > 30:
                m = min(8, (30 - 2) // 2)  # Adjust m if needed
                height = 2 * m + 2
                width = m
            
            # Choose colors for top row (m different colors, excluding 0)
            available_colors = list(range(1, 10))  # Colors 1-9
            top_row_colors = random.sample(available_colors, m)
            
            # Choose color_1 for second row (different from top row colors and not 0)
            remaining_colors = [c for c in available_colors if c not in top_row_colors]
            color_1 = random.choice(remaining_colors)
            
            gridvars = {
                'm': m,
                'top_row_colors': top_row_colors
            }
            
            taskvars = {
                'color_1': color_1
            }
            
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            
            return taskvars, gridvars, {
                'input': input_grid,
                'output': output_grid
            }
        
        # Generate training examples
        train_examples = []
        for _ in range(num_train):
            taskvars, gridvars, example = create_single_example()
            train_examples.append(example)
        
        # Generate test example
        test_taskvars, test_gridvars, test_example = create_single_example()
        test_examples = [test_example]
        
        # For template instantiation, we only need the consistent task variables
        # Since color_1 also varies, we'll use a representative value for templates
        template_taskvars = {
            'color_1': random.randint(1, 9)  # This is just for template rendering
        }
        
        return template_taskvars, {
            'train': train_examples,
            'test': test_examples
        }

