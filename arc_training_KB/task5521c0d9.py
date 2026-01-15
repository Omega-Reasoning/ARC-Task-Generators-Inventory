from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
import numpy as np
import random
from transformation_library import find_connected_objects

class Task5521c0d9Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are of size {vars['grid_size']}x{vars['grid_size']}.",
            "They contain exactly three rectangular blocks of colors {color('block_color1')}, {color('block_color2')}, and {color('block_color3')}.",
            "These blocks are always connected to the last edge of the grid, but their positions vary across examples.",
            "Each rectangular block is separated by the other by at least one empty (0) column.",
            "The lengths of the blocks within a grid are different and vary across examples.",
            "The length of each block should always be less than {vars['grid_size']//2}.",
            "The remaining cells are empty (0)."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is constructed by copying the input grid and moving the three colored blocks vertically upwards.",
            "The number of cells each block is moved matches the vertical length of that particular block.",
            "This transformation preserves the size and color of each block."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_grids(self):
        # Set task variables
        taskvars = {}
        # Ensure grid is big enough to accommodate our blocks
        taskvars['grid_size'] = random.randint(7, 30)  
        
        # Select three different colors for blocks
        all_colors = list(range(1, 10))  # Colors 1-9
        colors = random.sample(all_colors, 3)
        taskvars['block_color1'] = colors[0]
        taskvars['block_color2'] = colors[1]
        taskvars['block_color3'] = colors[2]
        
        # Generate train examples (3-4)
        num_train = random.randint(3, 4)
        train_examples = []
        
        for _ in range(num_train):
            # Randomize the order of colors for variety
            color_order = random.sample([taskvars['block_color1'], taskvars['block_color2'], taskvars['block_color3']], 3)
            gridvars = {'color_order': color_order}
            
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Generate test example
        color_order = random.sample([taskvars['block_color1'], taskvars['block_color2'], taskvars['block_color3']], 3)
        gridvars = {'color_order': color_order}
        
        test_input = self.create_input(taskvars, gridvars)
        test_output = self.transform_input(test_input, taskvars)
        
        test_examples = [{
            'input': test_input,
            'output': test_output
        }]
        
        return taskvars, {
            'train': train_examples,
            'test': test_examples
        }
    
    def create_input(self, taskvars, gridvars):
        grid_size = taskvars['grid_size']
        
        # Create empty grid
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Get color order for this grid
        color_order = gridvars['color_order']
        
        # Calculate maximum block length (less than half the grid)
        max_length = (grid_size // 2)
        
        # Ensure we have at least 5 different possible heights to sample from
        min_height = 2
        if max_length - min_height < 3:
            max_length = min_height + 3
        
        # Determine heights for the three blocks (ensuring they're different)
        heights = random.sample(range(min_height, max_length), 3)
        
        # Initialize grid with available columns
        available_cols = list(range(grid_size))
        blocks_placed = 0
        
        max_attempts = 50
        for attempt in range(max_attempts):
            if blocks_placed == 3:
                break
                
            # Pick the next color
            color = color_order[blocks_placed]
            height = heights[blocks_placed]
            
            # Determine a width (1-3 columns)
            max_possible_width = min(3, len(available_cols))
            if max_possible_width < 1:
                # Not enough columns, restart grid creation
                grid = np.zeros((grid_size, grid_size), dtype=int)
                available_cols = list(range(grid_size))
                blocks_placed = 0
                continue
                
            width = random.randint(1, max_possible_width)
            
            # Find valid starting positions (consecutive available columns)
            start_positions = []
            for i in range(len(available_cols) - width + 1):
                if all(c in available_cols for c in range(available_cols[i], available_cols[i] + width)):
                    start_positions.append(available_cols[i])
            
            if not start_positions:
                # No valid positions, try smaller width
                continue
                
            # Choose a random starting position
            start_col = random.choice(start_positions)
            
            # Create the rectangular block
            for col in range(start_col, start_col + width):
                for row in range(grid_size - height, grid_size):
                    grid[row, col] = color
            
            # Remove used columns and adjacent columns from available positions
            for col in range(max(0, start_col - 1), min(grid_size, start_col + width + 1)):
                if col in available_cols:
                    available_cols.remove(col)
            
            blocks_placed += 1
        
        # If we couldn't place all 3 blocks, try again
        if blocks_placed != 3:
            return self.create_input(taskvars, gridvars)
        
        # Verify we have exactly 3 blocks
        objects = find_connected_objects(grid, diagonal_connectivity=False, background=0)
        if len(objects) != 3:
            return self.create_input(taskvars, gridvars)
            
        return grid
    
    def transform_input(self, grid, taskvars):
        # Create a copy of the input grid
        output_grid = np.zeros_like(grid)
        
        # Find all blocks in the grid
        objects = find_connected_objects(grid, diagonal_connectivity=False, background=0)
        
        for obj in objects:
            # Get block properties
            rows, cols, _ = zip(*obj.cells)
            min_row = min(rows)
            max_row = max(rows)
            height = max_row - min_row + 1  # This is the height/length of the block
            color = next(iter(obj.colors))  # Get the color of the block
            
            # Move block upward by its height
            for r, c, _ in obj.cells:
                new_r = r - height
                if new_r >= 0:  # Ensure we stay within grid boundaries
                    output_grid[new_r, c] = color
        
        return output_grid

