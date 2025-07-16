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
            "All blocks have the same height (1 row) but different widths starting from 1."
        ]
        
        transformation_reasoning_chain = [
            "The output grid arranges all horizontal blocks in a right-aligned staircase pattern.",
            "The staircase pattern starts with width 1 at the top and increases by exactly 1 each row.",
            "Each block is placed in consecutive rows, right-aligned to the right edge of the grid.",
            "The staircase pattern ends at the last row with the longest block.",
            "The last (bottom) row contains the longest block and uses color {color('last_row_color')}."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars, gridvars):
        # Grid size and max width from gridvars (varies per grid)
        height = gridvars['height']
        width = gridvars['width']
        max_width = gridvars['max_width']
        
        grid = np.zeros((height, width), dtype=int)
        
        # Create blocks with widths from 1 to max_width (one for each width)
        num_blocks = max_width
        block_widths = list(range(1, max_width + 1))
        
        # Generate colors (excluding 0 which is background)
        available_colors = list(range(1, 10))
        block_colors = random.sample(available_colors, num_blocks)
        
        # Ensure the longest block gets the last_row_color
        longest_width_idx = block_widths.index(max_width)
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
        
        # Create a mapping from width to color
        width_to_color = {}
        for obj in objects.objects:
            width = obj.width
            color = list(obj.colors)[0]
            width_to_color[width] = color
        
        # Find the maximum width to determine staircase size
        max_width = max(width_to_color.keys())
        
        # Create the staircase pattern: width 1 at top, increasing by 1 each row
        # Place them consecutively ending at the last row
        start_row = rows - max_width
        
        for i in range(max_width):
            width = i + 1  # Width increases from 1 to max_width
            row_position = start_row + i
            
            # Get the color for this width
            if width in width_to_color:
                color = width_to_color[width]
            else:
                # If somehow this width doesn't exist, use a default color
                color = 1
            
            # Place the row right-aligned
            col_position = cols - width
            output_grid[row_position, col_position:col_position + width] = color
        
        return output_grid
    
    def create_grids(self):
        # Generate task variables (only last_row_color)
        last_row_color = random.randint(1, 9)
        
        taskvars = {
            'last_row_color': last_row_color
        }
        
        # Generate 3-5 training examples
        num_train_examples = random.randint(3, 5)
        train_examples = []
        
        for _ in range(num_train_examples):
            # Each grid can have different dimensions and max_width
            max_width = random.randint(5, 8)  # This determines the staircase size
            
            # Grid width equals max_width (most important requirement)
            width = max_width
            
            # Grid height should accommodate the staircase plus some extra space
            height = max_width + random.randint(2, 5)
            
            gridvars = {
                'height': height,
                'width': width,
                'max_width': max_width
            }
            
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Generate test example
        test_max_width = random.randint(5, 8)
        test_width = test_max_width
        test_height = test_max_width + random.randint(2, 5)
        
        test_gridvars = {
            'height': test_height,
            'width': test_width,
            'max_width': test_max_width
        }
        
        test_input = self.create_input(taskvars, test_gridvars)
        test_output = self.transform_input(test_input, taskvars)
        test_examples = [{
            'input': test_input,
            'output': test_output
        }]
        
        return taskvars, {
            'train': train_examples,
            'test': test_examples
        }

# Test code
if __name__ == "__main__":
    generator = Taskbeb8660cGenerator()
    taskvars, train_test_data = generator.create_grids()
    
    print("Task Variables:", taskvars)
    ARCTaskGenerator.visualize_train_test_data(train_test_data)