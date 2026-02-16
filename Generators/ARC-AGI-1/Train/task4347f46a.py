from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from Framework.input_library import create_object, retry, Contiguity
from Framework.transformation_library import find_connected_objects

class Task4347f46aGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['rows']}x{vars['cols']}.",
            "They contain several colored rectangular blocks, each with a unique color.",
            "Each colored block is completely separated from the others by empty (0) cells.",
            "Each block has a different size, with both length and width being greater than two."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid and identifying all rectangular blocks.",
            "Once identified, all interior cells of each rectangular block are removed.",
            "This transformation results in one-cell wide colored frames remaining in the grid."
        ]
        
        taskvars_definitions = {}
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self):
        rows = random.randint(6, 30)  # Grid rows
        cols = random.randint(6, 30)  # Grid columns
        num_train = random.randint(3, 4)  # Number of training examples

        taskvars = {
            'rows': rows,
            'cols': cols
        }

        used_num_rectangles = set()  # Track used numbers for training

        # Generate training examples
        train_examples = []
        for _ in range(num_train):
            available = [x for x in range(1, 6) if x not in used_num_rectangles]
            num_rectangles = random.choice(available)
            used_num_rectangles.add(num_rectangles)

            input_grid = self.create_input(taskvars, num_rectangles)
            output_grid = self.transform_input(input_grid, taskvars)
            train_examples.append({'input': input_grid, 'output': output_grid})

        # Generate test example with a number not in training set
        available_test = [x for x in range(1, 6) if x not in used_num_rectangles]
        num_rectangles_test = random.choice(available_test)

        test_input = self.create_input(taskvars, num_rectangles_test)
        test_output = self.transform_input(test_input, taskvars)
        test_examples = [{'input': test_input, 'output': test_output}]

        return taskvars, {
            'train': train_examples,
            'test': test_examples
        }

    
    def create_input(self, taskvars, num_rectangles):
        rows = taskvars['rows']
        cols = taskvars['cols']
        
        # Create an empty grid
        grid = np.zeros((rows, cols), dtype=int)
        
        # Available colors (1-9, avoiding 0 which is background)
        available_colors = list(range(1, 10))
        random.shuffle(available_colors)
        
        # Place rectangles one by one
        for _ in range(num_rectangles):
            if not available_colors:
                break
                
            color = available_colors.pop(0)
            
            # Try to place a rectangle with minimum width and height of 3
            success = self._place_rectangle(grid, color, min_size=3)
            
            # If we can't place any more rectangles, stop
            if not success:
                break
                
        return grid
    
    def _place_rectangle(self, grid, color, min_size=3, max_attempts=50):
        """Try to place a rectangle of given color on the grid"""
        rows, cols = grid.shape
        
        for _ in range(max_attempts):
            # Random rectangle dimensions (at least min_size x min_size)
            height = random.randint(min_size, min(rows//2, 10))
            width = random.randint(min_size, min(cols//2, 10))
            
            # Random top-left position
            top = random.randint(0, rows - height)
            left = random.randint(0, cols - width)
            
            # Check if we can place the rectangle here (with 1 cell buffer for separation)
            valid = True
            for r in range(max(0, top-1), min(rows, top+height+1)):
                for c in range(max(0, left-1), min(cols, left+width+1)):
                    # If this is outside our actual rectangle but inside the buffer zone
                    if (r < top or r >= top+height or c < left or c >= left+width):
                        # Make sure buffer zone is empty
                        if grid[r, c] != 0:
                            valid = False
                            break
                if not valid:
                    break
            
            if valid:
                # Place the rectangle
                for r in range(top, top+height):
                    for c in range(left, left+width):
                        grid[r, c] = color
                return True
                
        return False  # Couldn't place after max attempts
    
    def transform_input(self, grid, taskvars):
        # Create a copy of the input grid
        output = grid.copy()
        
        # Find all connected objects
        objects = find_connected_objects(grid, diagonal_connectivity=False, background=0, monochromatic=True)
        
        # Process each object
        for obj in objects:
            # Get the object's bounding box
            r_slice, c_slice = obj.bounding_box
            
            # For each colored object, we'll remove the interior cells
            # First, get the color of this object
            color = list(obj.colors)[0]  # We know it's monochromatic
            
            # Extract the subgrid for this object
            subgrid = grid[r_slice, c_slice]
            
            # Find interior cells (not on the edge)
            interior_mask = np.ones_like(subgrid, dtype=bool)
            interior_mask[0, :] = False  # Top edge
            interior_mask[-1, :] = False  # Bottom edge
            interior_mask[:, 0] = False  # Left edge
            interior_mask[:, -1] = False  # Right edge
            
            # Only apply mask to cells matching our color
            interior_mask = interior_mask & (subgrid == color)
            
            # Remove interior cells
            output[r_slice.start:r_slice.stop, c_slice.start:c_slice.stop][interior_mask] = 0
        
        return output

