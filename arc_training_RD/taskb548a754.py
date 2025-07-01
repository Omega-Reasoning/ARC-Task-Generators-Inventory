from arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from transformation_library import find_connected_objects, GridObjects, GridObject
from input_library import create_object, retry, random_cell_coloring
import numpy as np
import random

class BlockExtensionTaskGenerator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "The input grids vary in size, typically squares or near-square rectangles.",
            "Each grid contains a single block that holds a shape either a square or a rectangle.",
            "The block is colored with a two-tone pattern where one color for the boundary and a different color for the interior.",
            "Additionally, each grid contains exactly one pointer cell, marked with the color {color('pointer_color')}. This pointer is always positioned either horizontally or vertically aligned with the block and never diagonally."
        ]
        
        transformation_reasoning_chain = [
            "The output grid is a copy of the input grid.",
            "Each block uses its associated pointer (the anchor cell) to determine the direction of extension.",
            "The entire block then extends in the direction towards the pointer, stopping precisely at the pointer position.",
            "The block extends in a manner that resembles an enlarged replica of the original, preserving its shape and color pattern. However, the key factor guiding this extension is the pointer position, which serves as the reference point for both the direction and limit of the extension."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars, gridvars):
        """Create an input grid with a block and pointer."""
        grid_size = gridvars['grid_size']
        grid = np.zeros((grid_size, grid_size), dtype=int)
        
        # Use consistent pointer color from taskvars
        pointer_color = taskvars['pointer_color']
        
        # Generate random boundary and interior colors for this specific grid
        boundary_color = random.choice([c for c in range(1, 10) if c != pointer_color])
        interior_color = random.choice([c for c in range(1, 10) if c not in [pointer_color, boundary_color]])
        
        # Store colors in gridvars for transform_input to access
        gridvars['pointer_color'] = pointer_color
        gridvars['boundary_color'] = boundary_color
        gridvars['interior_color'] = interior_color
        
        # Create a block (rectangle or square)
        block_width = random.randint(3, min(6, grid_size // 2))
        block_height = random.randint(3, min(6, grid_size // 2))
        
        # Position the block randomly but ensure space for pointer
        max_row = grid_size - block_height - 1
        max_col = grid_size - block_width - 1
        block_row = random.randint(1, max(1, max_row))
        block_col = random.randint(1, max(1, max_col))
        
        # Fill the block area
        for r in range(block_row, block_row + block_height):
            for c in range(block_col, block_col + block_width):
                if (r == block_row or r == block_row + block_height - 1 or 
                    c == block_col or c == block_col + block_width - 1):
                    grid[r, c] = boundary_color
                else:
                    grid[r, c] = interior_color
        
        # Choose alignment direction
        direction = random.choice(['up', 'down', 'left', 'right'])
        
        if direction == 'up':
            # Place pointer above the block
            pointer_row = random.randint(0, max(0, block_row - 1))
            pointer_col = random.randint(block_col, block_col + block_width - 1)
        elif direction == 'down':
            # Place pointer below the block
            pointer_row = random.randint(min(grid_size - 1, block_row + block_height), grid_size - 1)
            pointer_col = random.randint(block_col, block_col + block_width - 1)
        elif direction == 'left':
            # Place pointer to the left of the block
            pointer_row = random.randint(block_row, block_row + block_height - 1)
            pointer_col = random.randint(0, max(0, block_col - 1))
        else:  # right
            # Place pointer to the right of the block
            pointer_row = random.randint(block_row, block_row + block_height - 1)
            pointer_col = random.randint(min(grid_size - 1, block_col + block_width), grid_size - 1)
        
        grid[pointer_row, pointer_col] = pointer_color
        
        return grid
    
    def transform_input(self, input_grid, taskvars, gridvars=None):
        """Transform input by extending block towards pointer."""
        output_grid = input_grid.copy()
        
        # Get colors from gridvars (set during create_input)
        if gridvars is None:
            # Fallback: use taskvars pointer_color and detect others
            pointer_color = taskvars['pointer_color']
            unique_colors = np.unique(input_grid)
            unique_colors = unique_colors[unique_colors != 0]  # Remove background
            remaining_colors = [c for c in unique_colors if c != pointer_color]
            boundary_color = remaining_colors[0] if remaining_colors else 1
            interior_color = remaining_colors[1] if len(remaining_colors) > 1 else 2
        else:
            pointer_color = gridvars['pointer_color']
            boundary_color = gridvars['boundary_color']
            interior_color = gridvars['interior_color']
        
        # Find pointer position
        pointer_pos = np.where(input_grid == pointer_color)
        if len(pointer_pos[0]) == 0:
            return output_grid
        pointer_row, pointer_col = pointer_pos[0][0], pointer_pos[1][0]
        
        # Find block boundaries
        block_positions = np.where((input_grid == boundary_color) | (input_grid == interior_color))
        if len(block_positions[0]) == 0:
            return output_grid
            
        min_row, max_row = min(block_positions[0]), max(block_positions[0])
        min_col, max_col = min(block_positions[1]), max(block_positions[1])
        
        # Extend block towards pointer
        if pointer_row < min_row:  # Pointer is above
            # Extend upward - create new rows
            new_min_row = pointer_row
            for r in range(new_min_row, min_row):
                for c in range(min_col, max_col + 1):
                    output_grid[r, c] = interior_color
            # Update boundaries for the extended block
            for c in range(min_col, max_col + 1):
                output_grid[new_min_row, c] = boundary_color  # New top boundary
                
        elif pointer_row > max_row:  # Pointer is below
            # Extend downward - create new rows
            new_max_row = pointer_row
            for r in range(max_row + 1, new_max_row + 1):
                for c in range(min_col, max_col + 1):
                    output_grid[r, c] = interior_color
            # Update boundaries for the extended block
            for c in range(min_col, max_col + 1):
                output_grid[new_max_row, c] = boundary_color  # New bottom boundary
                
        elif pointer_col < min_col:  # Pointer is to the left
            # Extend leftward - create new columns
            new_min_col = pointer_col
            for r in range(min_row, max_row + 1):
                for c in range(new_min_col, min_col):
                    output_grid[r, c] = interior_color
            # Update boundaries for the extended block
            for r in range(min_row, max_row + 1):
                output_grid[r, new_min_col] = boundary_color  # New left boundary
                
        elif pointer_col > max_col:  # Pointer is to the right
            # Extend rightward - create new columns
            new_max_col = pointer_col
            for r in range(min_row, max_row + 1):
                for c in range(max_col + 1, new_max_col + 1):
                    output_grid[r, c] = interior_color
            # Update boundaries for the extended block
            for r in range(min_row, max_row + 1):
                output_grid[r, new_max_col] = boundary_color  # New right boundary
        
        # Ensure proper boundary coloring for the entire extended block
        extended_positions = np.where((output_grid == boundary_color) | (output_grid == interior_color))
        if len(extended_positions[0]) > 0:
            final_min_row, final_max_row = min(extended_positions[0]), max(extended_positions[0])
            final_min_col, final_max_col = min(extended_positions[1]), max(extended_positions[1])
            
            # Set boundaries only on the edges
            for r in range(final_min_row, final_max_row + 1):
                for c in range(final_min_col, final_max_col + 1):
                    if (r == final_min_row or r == final_max_row or 
                        c == final_min_col or c == final_max_col):
                        output_grid[r, c] = boundary_color
                    else:
                        output_grid[r, c] = interior_color
        
        return output_grid
    
    def create_grids(self) -> tuple[dict[str, any], TrainTestData]:
        """Create train and test grids with consistent pointer color."""
        # Generate consistent pointer color for all grids
        pointer_color = random.randint(1, 9)
        
        # Store only the consistent pointer color in taskvars
        taskvars = {
            'pointer_color': pointer_color,
        }
        
        # Generate 3-5 training examples
        num_train_examples = random.randint(3, 5)
        train_examples = []
        
        # Generate grid sizes
        min_size = 8
        max_size = 15
        all_sizes = [random.randint(min_size, max_size) for _ in range(num_train_examples + 1)]
        
        # Create training examples
        for i in range(num_train_examples):
            gridvars = {'grid_size': all_sizes[i]}
            
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars, gridvars)
            
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Create test example
        test_gridvars = {'grid_size': all_sizes[-1]}
        test_input = self.create_input(taskvars, test_gridvars)
        test_output = self.transform_input(test_input, taskvars, test_gridvars)
        
        test_examples = [{
            'input': test_input,
            'output': test_output
        }]
        
        return taskvars, {
            'train': train_examples,
            'test': test_examples
        }

