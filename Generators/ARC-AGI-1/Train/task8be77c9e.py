from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.transformation_library import find_connected_objects, GridObject
from Framework.input_library import create_object, retry, Contiguity
import numpy as np
import random

class Task8be77c9eGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "The input grid is a square grid with dimension {vars['rows']} X {vars['rows']}.",
            "There is one 4-way object present in the input grid of color {color('in_color')}.",
            "The remaining cells are empty(0)"
        ]
        
        transformation_reasoning_chain = [
            "The output grid has the same width but twice the height of the input grid.",
            "First copy the input grid to the output grid.",
            "Then reflect the input grid along the x-axis at the bottom edge, the new reflected grid is stacked vertically along the bottom edge of the initial output grid."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self) -> tuple[dict[str, any], TrainTestData]:
        # Initialize task variables
        taskvars = {
            'rows': random.randint(3, 15),  # Using smaller grids for better visualization
            'in_color': random.randint(1, 9)  # Random color between 1-9
        }
        
        # Create 3-4 train examples and 1 test example
        num_train_examples = random.randint(3, 4)
        train_examples = []
        
        for _ in range(num_train_examples):
            input_grid = self.create_input(taskvars, {})
            output_grid = self.transform_input(input_grid, taskvars)
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Create test example
        test_input = self.create_input(taskvars, {})
        test_output = self.transform_input(test_input, taskvars)
        
        train_test_data = {
            'train': train_examples,
            'test': [{'input': test_input, 'output': test_output}]
        }
        
        return taskvars, train_test_data
    
    def create_input(self, taskvars: dict[str, any], gridvars: dict[str, any]) -> np.ndarray:
        rows = taskvars['rows']
        in_color = taskvars['in_color']
        
        # Create a grid with a random 4-way connected object
        def generate_valid_object():
            # Make sure the object doesn't fill too much of the grid (at most 75%)
            object_size = random.randint(max(1, rows), int(rows*rows*0.75))
            
            # Create object
            grid = np.zeros((rows, rows), dtype=int)
            
            # Create a randomly placed 4-way connected object
            obj = create_object(
                height=rows,
                width=rows,
                color_palette=in_color,
                contiguity=Contiguity.FOUR,
                background=0
            )
            
            # Ensure we don't have a full grid by checking if any background cells remain
            return obj if np.sum(obj == 0) > 0 else None
        
        # Try generating valid objects until we get one
        input_grid = retry(
            generate_valid_object,
            predicate=lambda x: x is not None and np.any(x != 0),
            max_attempts=100
        )
        
        return input_grid
    
    def transform_input(self, grid: np.ndarray, taskvars: dict[str, any]) -> np.ndarray:
        rows = taskvars['rows']
        
        # Create output grid with twice the height
        output = np.zeros((rows * 2, rows), dtype=int)
        
        # Copy input grid to the top part of output
        output[:rows, :] = grid
        
        # Reflect the input grid and place at bottom part
        reflected = np.flip(grid, axis=0)
        output[rows:, :] = reflected
        
        return output
