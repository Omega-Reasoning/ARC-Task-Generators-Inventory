from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from transformation_library import find_connected_objects, GridObject
from input_library import Contiguity, create_object, retry

class Task694f12f3Generator(ARCTaskGenerator):
    def __init__(self):
        # Initialize input reasoning chain
        input_reasoning_chain = [
            "Input grids are of size {vars['grid_size']}x{vars['grid_size']}.",
            "They contain exactly two {color('block_color')} rectangular blocks, with the remaining cells being empty (0).",
            "The blocks must be completely separated from each other and occupy more than two rows and columns each.",
            "The sizes of the blocks must be different within each grid and should vary across examples."
        ]
        
        # Initialize transformation reasoning chain
        transformation_reasoning_chain = [
            "The output grids are constructed by copying the input grids and identifying the larger and smaller {color('block_color')} rectangular blocks.",
            "Once identified, fill all interior cells of the larger block with {color('fill_color1')}, leaving the {color('block_color')} border intact.",
            "Similarly, fill all interior cells of the smaller block with {color('fill_color2')}, also preserving its {color('block_color')} border."
        ]
        
        # Call parent's init method with the reasoning chains
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self) -> tuple[dict[str, any], TrainTestData]:
        # Initialize task variables
        taskvars = {
            'grid_size': random.randint(10, 20),
            'block_color': random.randint(1, 9)
        }
        
        # Ensure fill colors are different from block color and each other
        available_colors = [c for c in range(1, 10) if c != taskvars['block_color']]
        fill_colors = random.sample(available_colors, 2)
        taskvars['fill_color1'] = fill_colors[0]
        taskvars['fill_color2'] = fill_colors[1]
        
        # Generate 3-4 training examples
        num_train_examples = random.randint(3, 4)
        
        train_data = []
        for _ in range(num_train_examples):
            # Create input grid with random block sizes
            input_grid = self.create_input(taskvars, {})
            # Transform input to output according to rules
            output_grid = self.transform_input(input_grid, taskvars)
            train_data.append({'input': input_grid, 'output': output_grid})
        
        # Generate 1 test example
        test_input = self.create_input(taskvars, {})
        test_output = self.transform_input(test_input, taskvars)
        
        # Return task variables and train/test data
        return taskvars, {
            'train': train_data,
            'test': [{'input': test_input, 'output': test_output}]
        }
    
    def create_input(self, taskvars: dict, gridvars: dict) -> np.ndarray:
        grid_size = taskvars['grid_size']
        block_color = taskvars['block_color']
        
        # Create empty grid
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Create the first block
        block1_height = random.randint(3, max(3, grid_size // 2))
        block1_width = random.randint(3, max(3, grid_size // 2))
        
        # Keep trying to place blocks until successful
        def generate_valid_grid():
            # Start with an empty grid
            grid = np.zeros((grid_size, grid_size), dtype=int)
            
            # Generate first block's dimensions
            block1_height = random.randint(3, max(3, grid_size // 2))
            block1_width = random.randint(3, max(3, grid_size // 2))
            
            # Generate second block's dimensions (ensure they're different)
            while True:
                block2_height = random.randint(3, max(3, grid_size // 2))
                block2_width = random.randint(3, max(3, grid_size // 2))
                # Ensure blocks have different area
                if block1_height * block1_width != block2_height * block2_width:
                    break
            
            # Place the first block
            row1 = random.randint(0, grid_size - block1_height)
            col1 = random.randint(0, grid_size - block1_width)
            grid[row1:row1+block1_height, col1:col1+block1_width] = block_color
            
            # Try to place the second block (with multiple attempts)
            for _ in range(50):  # Limit attempts
                row2 = random.randint(0, grid_size - block2_height)
                col2 = random.randint(0, grid_size - block2_width)
                
                # Check if blocks overlap
                if (row2 + block2_height <= row1 or row2 >= row1 + block1_height or
                    col2 + block2_width <= col1 or col2 >= col1 + block1_width):
                    # No overlap, place the second block
                    grid[row2:row2+block2_height, col2:col2+block2_width] = block_color
                    return grid
            
            # If we couldn't place the second block, raise an exception to trigger a retry
            raise ValueError("Could not place second block without overlap")
        
        # Generate grid with retry mechanism
        return retry(
            generate_valid_grid,
            lambda g: len(find_connected_objects(g, diagonal_connectivity=False, background=0)) == 2,
            max_attempts=100
        )
    
    def transform_input(self, grid: np.ndarray, taskvars: dict) -> np.ndarray:
        # Copy the input grid to avoid modifying it
        output_grid = grid.copy()
        
        # Get block color and fill colors
        block_color = taskvars['block_color']
        fill_color1 = taskvars['fill_color1']
        fill_color2 = taskvars['fill_color2']
        
        # Find all blocks in the grid
        objects = find_connected_objects(grid, diagonal_connectivity=False, background=0)
        
        # Sort blocks by size (largest first)
        objects = objects.sort_by_size(reverse=True)
        
        # Process each block
        for i, obj in enumerate(objects):
            # Get block dimensions
            box = obj.bounding_box
            block_array = grid[box[0], box[1]]
            h, w = block_array.shape
            
            # Create array for interior (all cells except border)
            interior = np.zeros_like(block_array)
            interior[1:h-1, 1:w-1] = 1  # Mark interior cells
            
            # Apply appropriate fill color for interior
            fill_color = fill_color1 if i == 0 else fill_color2
            
            # Fill interior of block in output grid
            for r in range(box[0].start, box[0].stop):
                for c in range(box[1].start, box[1].stop):
                    # Check if it's an interior cell
                    r_rel = r - box[0].start
                    c_rel = c - box[1].start
                    
                    if (1 <= r_rel < h-1 and 1 <= c_rel < w-1 and 
                        output_grid[r, c] == block_color):
                        output_grid[r, c] = fill_color
        
        return output_grid

