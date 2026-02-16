from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from Framework.input_library import random_cell_coloring

class Task46442a0eGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['grid_size']}x{vars['grid_size']}.",
            "They contain a completely filled grid with colored (1-9) cells that sometimes form square and rectangular blocks.",
            "Each input grid uses two or three different colors, with the colors varying across examples.",
            "There are no empty (0) cells in the input grids."
        ]
        
        transformation_reasoning_chain = [
            "The output grids are of size {2*vars['grid_size']}x{2*vars['grid_size']}.",
            "They are constructed by rotating the input grid by 90, 180, and 270 degrees clockwise and placing each rotated version in different quadrants of the output grid.",
            "The original input grid is placed in the first quadrant, the 90-degree rotated version in the second quadrant, the 180-degree rotated version in the third quadrant, and the 270-degree rotated version in the fourth quadrant."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self) -> tuple[dict[str, any], TrainTestData]:
        # Define task variables
        taskvars = {
            'grid_size': random.randint(3, 15)
        }
        
        # Create train examples
        train_examples = []
        
        # Create a special training example with a rectangular block in top-left corner
        special_example1 = self._create_special_example1(taskvars)
        train_examples.append(special_example1)
        
        # Create a special training example with a rectangular block in bottom-right corner
        special_example2 = self._create_special_example2(taskvars)
        train_examples.append(special_example2)
        
        # Create additional random training examples
        num_additional = random.randint(1, 3)
        for _ in range(num_additional):
            input_grid = self.create_input(taskvars, {})
            output_grid = self.transform_input(input_grid, taskvars)
            train_examples.append({'input': input_grid, 'output': output_grid})
        
        # Create test example
        test_input = self.create_input(taskvars, {})
        test_output = self.transform_input(test_input, taskvars)
        
        return taskvars, {
            'train': train_examples,
            'test': [{'input': test_input, 'output': test_output}]
        }
    
    def _create_special_example1(self, taskvars):
        """Create an example with a rectangular block in top-left corner"""
        grid_size = taskvars['grid_size']
        input_grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Choose two distinct colors
        colors = random.sample(range(1, 10), 2)
        
        # Create a (grid_size-1)×(grid_size-1) block in top-left corner
        input_grid[:grid_size-1, :grid_size-1] = colors[0]
        
        # Fill the remaining cells with the second color
        input_grid[grid_size-1, :] = colors[1]
        input_grid[:grid_size-1, grid_size-1] = colors[1]
        
        output_grid = self.transform_input(input_grid, taskvars)
        return {'input': input_grid, 'output': output_grid}
    
    def _create_special_example2(self, taskvars):
        """Create an example with a rectangular block in bottom-right corner"""
        grid_size = taskvars['grid_size']
        input_grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Choose three distinct colors
        colors = random.sample(range(1, 10), 3)
        
        # Fill first row with one color
        input_grid[0, :] = colors[0]
        
        # Fill first column with another color (except the first cell which is already colored)
        input_grid[1:, 0] = colors[1]
        
        # Create a (grid_size-1)×(grid_size-1) block in bottom-right corner
        input_grid[1:, 1:] = colors[2]
        
        output_grid = self.transform_input(input_grid, taskvars)
        return {'input': input_grid, 'output': output_grid}
    
    def create_input(self, taskvars, gridvars):
        grid_size = taskvars['grid_size']
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Decide how many colors to use (2 or 3)
        num_colors = random.randint(2, 3)
        colors = random.sample(range(1, 10), num_colors)
        
        # Create a grid with color blocks
        self._create_colored_blocks(grid, colors)
        
        return grid
    
    def _create_colored_blocks(self, grid, colors):
        """Helper method to create random colored blocks in the grid"""
        height, width = grid.shape
        
        # Start with a completely random filling
        for r in range(height):
            for c in range(width):
                grid[r, c] = random.choice(colors)
        
        # Create 1-3 rectangular blocks of random colors
        num_blocks = random.randint(1, 3)
        
        for _ in range(num_blocks):
            # Choose random position and size for the block
            block_height = random.randint(1, height-1)
            block_width = random.randint(1, width-1)
            row_start = random.randint(0, height - block_height)
            col_start = random.randint(0, width - block_width)
            
            # Fill the block with a random color
            color = random.choice(colors)
            grid[row_start:row_start+block_height, col_start:col_start+block_width] = color
    
    def transform_input(self, grid, taskvars):
        grid_size = taskvars['grid_size']
        output_size = 2 * grid_size
        output_grid = np.zeros((output_size, output_size), dtype=int)
        
        # Place original grid in first quadrant (top-left)
        output_grid[:grid_size, :grid_size] = grid
        
        # Place 90° clockwise rotated grid in second quadrant (top-right)
        rot90 = np.rot90(grid, k=3)  # k=3 for 90° clockwise (k=1 is 90° counterclockwise)
        output_grid[:grid_size, grid_size:] = rot90
        
        # Place 180° clockwise rotated grid in third quadrant (bottom-right)
        rot180 = np.rot90(grid, k=2)  # k=2 for 180°
        output_grid[grid_size:, grid_size:] = rot180
        
        # Place 270° clockwise rotated grid in fourth quadrant (bottom-left)
        rot270 = np.rot90(grid, k=1)  # k=1 for 90° counterclockwise = 270° clockwise
        output_grid[grid_size:, :grid_size] = rot270
        
        return output_grid

