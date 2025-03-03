from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from input_library import create_object, Contiguity
from transformation_library import find_connected_objects

class Task3af2c5a8Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['rows']}x{vars['cols']}.",
            "They contain single-colored cells and empty (0) cells.",
            "The single-colored cells form an 8-way connected object in the grid.",
            "The color of the object varies across examples."

        ]
        
        transformation_reasoning_chain = [
            "Output grids are of size {2 * vars['rows']}x{2 * vars['cols']}.",
            "They are constructed by first flipping the original {vars['rows']}x{vars['cols']} grid along the horizontal axis, doubling the height to {2 * vars['rows']}x{vars['cols']}.",
            "Next, the newly formed {2 * vars['rows']}x{vars['cols']} grid is mirrored along the vertical axis, doubling the width to {2 * vars['rows']}x{2 * vars['cols']}."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self):
        # Initialize task variables (dimensions)
        rows = random.randint(5, 10)
        cols = random.randint(5, 10)
        
        taskvars = {
            'rows': rows,
            'cols': cols
        }
        
        # Generate 3-5 training examples and 1 test example
        train_count = random.randint(3, 5)
        train_examples = []
        
        for _ in range(train_count):
            # Create a new input grid
            input_grid = self.create_input(taskvars, {})
            # Transform the input grid to get the output grid
            output_grid = self.transform_input(input_grid, taskvars)
            # Add the pair to the training examples
            train_examples.append({'input': input_grid, 'output': output_grid})
        
        # Create test example
        test_input = self.create_input(taskvars, {})
        test_output = self.transform_input(test_input, taskvars)
        
        # Return task variables and train/test data
        return taskvars, {
            'train': train_examples,
            'test': [{'input': test_input, 'output': test_output}]
        }
    
    def create_input(self, taskvars, gridvars):
        rows = taskvars['rows']
        cols = taskvars['cols']
        
        # Create a blank grid
        grid = np.zeros((rows, cols), dtype=int)
        
        # Choose a random color between 1 and 9
        color = random.randint(1, 9)
        
        # Create a connected object with the chosen color
        object_matrix = create_object(
            height=rows,
            width=cols,
            color_palette=color,
            contiguity=Contiguity.EIGHT,
            background=0
        )
        
        # Ensure the object has a reasonable size (not too small, not too large)
        min_size = max(rows * cols // 10, 3)  # At least 10% of grid or 3 cells
        max_size = rows * cols // 2  # At most 50% of grid
        
        # Get size of the object
        object_size = np.count_nonzero(object_matrix)
        
        # If object is too small or too large, retry
        if object_size < min_size or object_size > max_size:
            return self.create_input(taskvars, gridvars)
        
        # Assign the object matrix to the grid
        grid = object_matrix
        
        return grid
    
    def transform_input(self, grid, taskvars):
        rows = taskvars['rows']
        cols = taskvars['cols']
        
        # Create a larger output grid (2x the size in both dimensions)
        output = np.zeros((2 * rows, 2 * cols), dtype=int)
        
        # Step 1: Copy the original grid to the top-left quadrant
        output[:rows, :cols] = grid
        
        # Step 2: Flip along horizontal axis for bottom-left quadrant
        output[rows:2*rows, :cols] = np.flipud(grid)
        
        # Step 3: Mirror both quadrants along vertical axis
        output[:, cols:2*cols] = np.fliplr(output[:, :cols])
        
        return output

