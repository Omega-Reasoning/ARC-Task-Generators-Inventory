from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
from Framework.transformation_library import find_connected_objects
from Framework.input_library import create_object, retry, Contiguity
import random

class Taskd0f5fe59Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids can have different sizes.",
            "Each input grid contains n (a positive integer greater than 1) objects, where each object is made of 4-way connected cells of {color('object_color')} color. The remaining cells are empty (0).",
            "n varies for each input grid.",
            "The objects are of {color('object_color')} color.",
            "Each object is surrounded by empty (0) cells to ensure it does not touch other objects or the edges of the grid.",
            "The objects have different sizes.",
            "The size of each object is greater than 1."
        ]

        transformation_reasoning_chain = [
            "The output grid is always smaller than the input grid.",
            "It is constructed by first counting the number of {color('object_color')} objects in the input grid, let this number be n.",
            "Then, initialize an empty (0) grid of size n × n and completely fill in the main diagonal (from top left to bottom right) with {color('object_color')} color.",
            "All remaining cells in the output grid remain empty (0)."
        ]

        taskvars_definitions = {'object_color': random.randint(1, 9)}
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)

    def create_input(self, taskvars, gridvars):
        # Define grid parameters
        grid_size = random.randint(5, 30) 
        object_color = taskvars['object_color']
        num_objects = random.randint(2, 8)  # 2 to 8 objects
        
        # Create empty grid
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Place objects in the grid
        for _ in range(num_objects):
            self._place_random_object(grid, object_color)
            
        return grid
    
    def _place_random_object(self, grid, color):
        def generate_object():
            # Create a small object with random size between 2 and 6 cells
            object_size = random.randint(2, 6)
            
            # Start with a random cell
            grid_size = grid.shape[0]
            start_row = random.randint(1, grid_size - 2)  # Avoid edges
            start_col = random.randint(1, grid_size - 2)  # Avoid edges
            
            # If the starting cell or its surroundings are already occupied, try again
            for r in range(max(0, start_row-1), min(grid_size, start_row+2)):
                for c in range(max(0, start_col-1), min(grid_size, start_col+2)):
                    if grid[r, c] != 0:
                        return None  # Cell or its surroundings occupied
            
            # Start building the object
            cells = [(start_row, start_col)]
            grid[start_row, start_col] = color
            
            # Try to grow the object up to desired size
            attempts = 0
            while len(cells) < object_size and attempts < 20:
                attempts += 1
                
                # Pick a random cell from existing cells
                base_row, base_col = random.choice(cells)
                
                # Try to add an adjacent cell
                directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
                random.shuffle(directions)
                
                for dr, dc in directions:
                    new_row, new_col = base_row + dr, base_col + dc
                    
                    # Check if the new cell is inside the grid and empty
                    if (0 < new_row < grid_size-1 and 0 < new_col < grid_size-1 and 
                        grid[new_row, new_col] == 0):
                        
                        # Check if adding this cell would maintain the isolation requirement
                        isolated = True
                        for r in range(max(0, new_row-1), min(grid_size, new_row+2)):
                            for c in range(max(0, new_col-1), min(grid_size, new_col+2)):
                                # Skip the cells that are part of the current object
                                if (r, c) in cells or (r == new_row and c == new_col):
                                    continue
                                if grid[r, c] != 0:
                                    isolated = False
                                    break
                            if not isolated:
                                break
                        
                        if isolated:
                            cells.append((new_row, new_col))
                            grid[new_row, new_col] = color
                            break
            
            return cells if len(cells) >= 2 else None
        
        # Try to place an object until successful
        max_attempts = 50
        for _ in range(max_attempts):
            result = generate_object()
            if result is not None:
                return result
        
        # If we couldn't place an object after maximum attempts, just continue
        return None
    
    def transform_input(self, grid, taskvars):
        object_color = taskvars['object_color']
        
        # Find all objects of the specified color
        objects = find_connected_objects(grid, diagonal_connectivity=False, background=0)
        objects_of_color = objects.with_color(object_color)
        
        # Count the number of objects
        n = len(objects_of_color)
        
        # Create an n x n grid with the diagonal filled
        output = np.zeros((n, n), dtype=int)
        
        # Fill the main diagonal
        for i in range(n):
            output[i, i] = object_color
            
        return output
    
    def create_grids(self):
        # Create task vars
        taskvars = {'object_color': random.randint(1, 9)}
        
        # Determine number of training examples
        num_train = random.randint(3, 6)
        
        # Create training examples
        train_examples = []
        for _ in range(num_train):
            input_grid = self.create_input(taskvars, {})
            output_grid = self.transform_input(input_grid, taskvars)
            train_examples.append({'input': input_grid, 'output': output_grid})
        
        # Create test example
        test_input = self.create_input(taskvars, {})
        test_output = self.transform_input(test_input, taskvars)
        test_examples = [{'input': test_input, 'output': test_output}]
        
        return taskvars, {'train': train_examples, 'test': test_examples}
