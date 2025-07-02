from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects, GridObjects, GridObject
from input_library import retry, random_cell_coloring
import numpy as np
import random

class Taskbeb8660cGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "The input grids contain horizontal blocks of various colors and widths.",
            "Each horizontal block is a solid rectangle spanning different widths.",
            "The blocks are placed randomly in the input grid with gaps between them.",
            "All blocks have the same height (1 row) but different widths."
        ]
        
        transformation_reasoning_chain = [
            "The output grid arranges all horizontal blocks in a right-aligned staircase pattern.",
            "Blocks are sorted by width in ascending order (shortest to longest).",
            "Each block is placed in consecutive rows, right-aligned to the right edge of the grid.",
            "The staircase pattern ends at the last row of the grid.",
            "The last (bottom) row contains the longest block and uses color {color('last_row_color')}."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars):
        # Grid size
        height = random.randint(8, 12)
        width = random.randint(10, 15)
        
        grid = np.zeros((height, width), dtype=int)
        
        # Create 3-6 horizontal blocks with different widths and colors
        num_blocks = random.randint(3, 6)
        
        # Generate unique widths for each block
        min_width = 2
        max_width = min(width - 1, 8)
        
        # Create a list of possible widths and sample from it
        possible_widths = list(range(min_width, max_width + 1))
        if len(possible_widths) < num_blocks:
            num_blocks = len(possible_widths)
        
        block_widths = random.sample(possible_widths, num_blocks)
        
        # Generate colors (excluding 0 which is background)
        available_colors = list(range(1, 10))
        block_colors = random.sample(available_colors, num_blocks)
        
        # Ensure the longest block gets the last_row_color
        longest_width_idx = block_widths.index(max(block_widths))
        block_colors[longest_width_idx] = taskvars['last_row_color']
        
        # Place blocks randomly in the grid
        placed_blocks = []
        
        for width, color in zip(block_widths, block_colors):
            # Try to place the block
            max_attempts = 50
            placed = False
            
            for _ in range(max_attempts):
                # Random position
                row = random.randint(0, height - 1)
                col = random.randint(0, width - width)  # Ensure block fits
                
                # Check if position is free
                if np.all(grid[row, col:col + width] == 0):
                    # Place the block
                    grid[row, col:col + width] = color
                    placed_blocks.append((width, color, row, col))
                    placed = True
                    break
            
            # If we couldn't place the block, try a simpler placement
            if not placed:
                for row in range(height):
                    for col in range(width - width + 1):
                        if np.all(grid[row, col:col + width] == 0):
                            grid[row, col:col + width] = color
                            placed_blocks.append((width, color, row, col))
                            placed = True
                            break
                    if placed:
                        break
        
        return grid
    
    def transform_input(self, grid, taskvars):
        # Create output grid of same size
        output_grid = np.zeros_like(grid)
        rows, cols = grid.shape
        
        # Find all horizontal blocks
        objects = find_connected_objects(grid, diagonal_connectivity=False, background=0)
        
        if len(objects.objects) == 0:
            return output_grid
        
        # Sort objects by width in ascending order (shortest to longest)
        objects_by_width = sorted(objects.objects, key=lambda obj: obj.width)
        
        # Calculate starting row position to place the staircase
        # We want to place them consecutively ending at the last row
        num_objects = len(objects_by_width)
        start_row = rows - num_objects
        
        # Place rows in ascending order by width, right-aligned
        for i, obj in enumerate(objects_by_width):
            color = list(obj.colors)[0]
            width = obj.width
            row_position = start_row + i
            
            # Place the row right-aligned
            col_position = cols - width
            output_grid[row_position, col_position:col_position + width] = color
        
        return output_grid
    
    def create_grids(self):
        # Generate task variables
        last_row_color = random.randint(1, 9)
        
        taskvars = {
            'last_row_color': last_row_color
        }
        
        # Generate 3-5 training examples
        num_train_examples = random.randint(3, 5)
        train_examples = []
        
        for _ in range(num_train_examples):
            input_grid = self.create_input(taskvars)
            output_grid = self.transform_input(input_grid, taskvars)
            train_examples.append(GridPair(input=input_grid, output=output_grid))
        
        # Generate test example
        test_input = self.create_input(taskvars)
        test_output = self.transform_input(test_input, taskvars)
        test_examples = [GridPair(input=test_input, output=test_output)]
        
        return taskvars, TrainTestData(train=train_examples, test=test_examples)

