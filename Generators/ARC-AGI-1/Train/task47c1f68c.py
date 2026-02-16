from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from Framework.input_library import create_object, Contiguity, retry
from Framework.transformation_library import find_connected_objects, GridObject, BorderBehavior

class Task47c1f68cGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['grid_size']}x{vars['grid_size']}.",
            "They contain a cross shape formed by filling the entire middle row and middle column with the same-colored cells.",
            "The cross divides the grid into four quadrants, with an object placed in the top-left quadrant, connected to either the top or left edge of the grid.",
            "This object is made of 8-way connected cells in a color different from the cross shape.",
            "The object in the first quadrant must not overlap with the cells of the cross shape.",
            "Each grid uses two different colors: one for the cross shape and one for the object, with the colors varying across examples."
        ]

        
        transformation_reasoning_chain = [
            "The output grids are of size {vars['grid_size']-1}x{vars['grid_size']-1}.",
            "Reflect the colored object in the top-left quadrant to the second quadrant, using the fully colored middle column as the line of reflection.",
            "Then, both objects from the first and second quadrants are reflected vertically downward, using the fully colored middle row as the line of reflection.",
            "Finally, remove the completely colored middle row and column changing the grid size from {vars['grid_size']}x{vars['grid_size']} to {vars['grid_size']-1}x{vars['grid_size']-1}."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self):
        # Choose grid size (odd number between 5 and 19 for variability)
        grid_size = random.choice([5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29])
        
        taskvars = {
            'grid_size': grid_size
        }
        
        
        num_train_examples = random.randint(3, 4)
        
        train_examples = []
        for _ in range(num_train_examples):
            # Choose colors for cross and object for each example
            cross_color = random.randint(1, 9)
            while True:
                object_color = random.randint(1, 9)
                if object_color != cross_color:
                    break
                    
            gridvars = {
                'cross_color': cross_color,
                'object_color': object_color
            }
            
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            train_examples.append({'input': input_grid, 'output': output_grid})
        
        # Generate 1 test example
        test_cross_color = random.randint(1, 9)
        while True:
            test_object_color = random.randint(1, 9)
            if test_object_color != test_cross_color:
                break
                
        test_gridvars = {
            'cross_color': test_cross_color,
            'object_color': test_object_color
        }
        
        test_input = self.create_input(taskvars, test_gridvars)
        test_output = self.transform_input(test_input, taskvars)
        test_examples = [{'input': test_input, 'output': test_output}]
        
        return taskvars, {'train': train_examples, 'test': test_examples}
    
    def create_input(self, taskvars, gridvars):
        grid_size = taskvars['grid_size']
        cross_color = gridvars['cross_color']
        object_color = gridvars['object_color']
        
        # Create empty grid
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Add cross in the middle
        mid = grid_size // 2
        grid[mid, :] = cross_color  # Middle row
        grid[:, mid] = cross_color  # Middle column
        
        # Create the top-left quadrant
        quadrant = np.zeros((mid, mid), dtype=int)
        
        # Generate object in the quadrant connected to top or left edge
        def generate_valid_object():
            # Create a candidate object in the exact quadrant size
            obj_grid = create_object(
                height=mid, 
                width=mid, 
                color_palette=object_color,
                contiguity=Contiguity.EIGHT,
                background=0
            )
            
            # Ensure object connects to top or left edge
            objects = find_connected_objects(obj_grid, diagonal_connectivity=True)
            
            if len(objects) == 0:
                return None  # No objects found
                
            for obj in objects:
                # Check if any cells of the object are at the top or left edge
                for r, c, _ in obj.cells:
                    if r == 0 or c == 0:
                        return obj_grid  # Return the entire grid with the valid object
            
            return None  # No object connects to edge
        
        # Try to generate a valid object
        quadrant = retry(
            generate_valid_object,
            lambda x: x is not None,
            max_attempts=100
        )
        
        # Place the object in the top-left quadrant
        grid[:mid, :mid] = quadrant
        
        return grid
    
    def transform_input(self, grid, taskvars):
        grid_size = taskvars['grid_size']
        
        # Create a copy of the input grid
        output_grid = grid.copy()
        
        # Find the middle indices for the cross
        mid = grid_size // 2
        
        # Get the cross color from the middle cell
        cross_color = grid[mid, mid]
        
        # Find the object in the top-left quadrant
        top_left_mask = np.zeros_like(grid, dtype=bool)
        top_left_mask[:mid, :mid] = True
        top_left_quadrant = grid.copy()
        # Zero out everything except the top-left quadrant
        top_left_quadrant[~top_left_mask] = 0
        
        # Find the connected objects in the top-left quadrant
        objects = find_connected_objects(top_left_quadrant, diagonal_connectivity=True, background=0)
        if len(objects) > 0:
            # Get the original object
            original_object = objects[0]
            
            # Reflect horizontally (to quadrant 2)
            for r, c, color in original_object.cells:
                # Calculate distance from the middle column
                dist_from_mid = mid - c
                # New column position after reflection
                new_c = mid + dist_from_mid
                # Add the reflected cell
                if 0 <= new_c < grid_size:
                    output_grid[r, new_c] = color
            
            # Reflect both objects vertically (to quadrants 3 and 4)
            for r in range(mid):
                for c in range(grid_size):
                    if output_grid[r, c] != 0 and output_grid[r, c] != cross_color:
                        # Calculate distance from middle row
                        dist_from_mid = mid - r
                        # New row position after reflection
                        new_r = mid + dist_from_mid
                        # Add the reflected cell
                        if 0 <= new_r < grid_size:
                            output_grid[new_r, c] = output_grid[r, c]
        
        # Remove the middle row and column
        # We'll create a new grid of size grid_size-1 x grid_size-1
        final_grid = np.zeros((grid_size-1, grid_size-1), dtype=int)
        
        # Copy each quadrant to final grid, skipping the middle row and column
        for r in range(grid_size-1):
            r_src = r if r < mid else r + 1
            for c in range(grid_size-1):
                c_src = c if c < mid else c + 1
                final_grid[r, c] = output_grid[r_src, c_src]
        
        return final_grid