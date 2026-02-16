from Framework.arc_task_generator import ARCTaskGenerator, GridPair, TrainTestData
from Framework.transformation_library import find_connected_objects, GridObjects
from Framework.input_library import create_object, random_cell_coloring, retry
import numpy as np
import random

class Task6f473927Generator(ARCTaskGenerator):
    def __init__(self):
        input_reasoning_chain = [
            "Input grids are mostly rectangles and sometimes squares of different sizes.",
            "Each input grid creates a very unusual pattern of {color('pattern_color')}, for which the pattern strictly starts from either the last column or the first column."
        ]
        
        transformation_reasoning_chain = [
            "The output grid sizes are copied from input where the row size remains same, but column size doubles.",
            "The transformation is that, consider it as a super mirror, when the input grid is super imposed on the other side of the grid, the cells which are having the pattern color will get the color of empty cell(0), and rest all of the cells must get the color {color('fill_color')}. So basically this super mirror captures the pattern alone from the other side and gives another color to the entire grid excluding the pattern, and the pattern gets empty cells color(0).",
            "This part is very crucial, so check from which column the pattern was being created in the input grid. If the pattern started from the last column, then you have to add the superimposed block on the right and vice versa."
        ]
        
        super().__init__(input_reasoning_chain, transformation_reasoning_chain)
    
    def create_input(self, taskvars, gridvars):
        """Create input grid with pattern starting strictly from first or last column."""
        grid_height = gridvars['grid_height']
        grid_width = gridvars['grid_width']
        pattern_color = taskvars['pattern_color']
        pattern_side = gridvars['pattern_side']  # 'left' or 'right'
        
        # Create base grid with background
        grid = np.zeros((grid_height, grid_width), dtype=int)
        
        # Create pattern that starts STRICTLY from specified column
        if pattern_side == 'left':
            # Pattern starts STRICTLY from first column (column 0)
            start_col = 0
            # Ensure we have at least one pattern cell in the first column
            start_row = random.randint(0, grid_height - 1)
            grid[start_row, start_col] = pattern_color
            
            # Add more pattern cells, but ensure they don't appear in the last column
            forbidden_col = grid_width - 1
        else:
            # Pattern starts STRICTLY from last column
            start_col = grid_width - 1
            # Ensure we have at least one pattern cell in the last column
            start_row = random.randint(0, grid_height - 1)
            grid[start_row, start_col] = pattern_color
            
            # Add more pattern cells, but ensure they don't appear in the first column
            forbidden_col = 0
        
        # Randomly add more pattern cells
        pattern_density = random.uniform(0.15, 0.4)
        num_pattern_cells = max(1, int(grid_height * grid_width * pattern_density))
        
        # Add pattern cells, avoiding the forbidden column
        cells_added = 1  # We already added one
        attempts = 0
        max_attempts = num_pattern_cells * 20
        
        while cells_added < num_pattern_cells and attempts < max_attempts:
            attempts += 1
            
            row = random.randint(0, grid_height - 1)
            col = random.randint(0, grid_width - 1)
            
            # Skip the forbidden column to ensure pattern starts from only one end
            if col == forbidden_col:
                continue
                
            if grid[row, col] == 0:  # Only add to empty cells
                grid[row, col] = pattern_color
                cells_added += 1
        
        return grid
    
    def transform_input(self, grid, taskvars):
        """Transform input using super mirror logic."""
        # Dynamically detect pattern color from the input grid
        unique_colors = np.unique(grid)
        non_zero_colors = unique_colors[unique_colors != 0]
        
        if len(non_zero_colors) == 0:
            # No pattern found, use default
            pattern_color = taskvars['pattern_color']
        else:
            # Use the first non-zero color as pattern color
            pattern_color = non_zero_colors[0]
        
        # Use the fill_color for the super mirror transformation
        fill_color = taskvars['fill_color']
        
        height, width = grid.shape
        
        # Create output grid with double width
        output_grid = np.zeros((height, width * 2), dtype=int)
        
        # Check if pattern starts from first or last column
        first_col_has_pattern = np.any(grid[:, 0] == pattern_color)
        last_col_has_pattern = np.any(grid[:, -1] == pattern_color) 
        
        # Since we ensure strict starting, this should be clear
        if first_col_has_pattern:
            pattern_starts_from_first = True
        elif last_col_has_pattern:
            pattern_starts_from_first = False
        else:
            # Fallback (should not happen with strict generation)
            left_half_pattern = np.sum(grid[:, :width//2] == pattern_color)
            right_half_pattern = np.sum(grid[:, width//2:] == pattern_color)
            pattern_starts_from_first = left_half_pattern >= right_half_pattern
        
        if pattern_starts_from_first:
            # Pattern starts from first column (left)
            # Place original input on the right half
            output_grid[:, width:] = grid
            
            # Create horizontally mirrored super mirror on the left half
            for r in range(height):
                for c in range(width):
                    # Mirror horizontally: column c becomes column (width - 1 - c)
                    mirrored_c = width - 1 - c
                    if grid[r, c] == pattern_color:
                        # Pattern cells become background (0)
                        output_grid[r, mirrored_c] = 0
                    else:
                        # Background cells become fill_color
                        output_grid[r, mirrored_c] = fill_color
        else:
            # Pattern starts from last column (right)
            # Place original input on the left half
            output_grid[:, :width] = grid
            
            # Create horizontally mirrored super mirror on the right half
            for r in range(height):
                for c in range(width):
                    # Mirror horizontally: column c becomes column (width - 1 - c)
                    mirrored_c = width + (width - 1 - c)
                    if grid[r, c] == pattern_color:
                        # Pattern cells become background (0)
                        output_grid[r, mirrored_c] = 0
                    else:
                        # Background cells become fill_color
                        output_grid[r, mirrored_c] = fill_color
        
        return output_grid
    
    def create_grids(self):
        """Create train and test grids with consistent variables."""
        # Generate random colors ensuring they're all different
        pattern_color = random.randint(1, 9)
        remaining_colors = [c for c in range(1, 10) if c != pattern_color]
        output_color = random.choice(remaining_colors)
        remaining_colors.remove(output_color)
        fill_color = random.choice(remaining_colors)
        
        # Store task variables
        taskvars = {
            'pattern_color': pattern_color,
            'output_color': output_color,
            'fill_color': fill_color,
        }
        
        # Generate 3-5 training examples
        num_train_examples = random.randint(3, 5)
        train_examples = []
        
        # Create training examples
        for i in range(num_train_examples):
            # Generate grid dimensions
            grid_height = random.randint(3, 6)
            grid_width = random.randint(3, 6)
            pattern_side = random.choice(['left', 'right'])
            
            gridvars = {
                'grid_height': grid_height,
                'grid_width': grid_width,
                'pattern_side': pattern_side
            }
            
            input_grid = self.create_input(taskvars, gridvars)
            output_grid = self.transform_input(input_grid, taskvars)
            
            train_examples.append({
                'input': input_grid,
                'output': output_grid
            })
        
        # Create test example
        test_height = random.randint(3, 6)
        test_width = random.randint(3, 6)
        test_pattern_side = random.choice(['left', 'right'])
        
        test_gridvars = {
            'grid_height': test_height,
            'grid_width': test_width,
            'pattern_side': test_pattern_side
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

# Test the generator
if __name__ == "__main__":
    generator = SuperMirrorTaskGenerator()
    taskvars, train_test_data = generator.create_grids()
    
    print("Task Variables:", taskvars)
    print(f"Number of train examples: {len(train_test_data['train'])}")
    print(f"Number of test examples: {len(train_test_data['test'])}")
    
    # Visualize first training example
    if train_test_data['train']:
        first_example = train_test_data['train'][0]
        print("\nFirst training example:")
        print("Input shape:", first_example['input'].shape)  
        print("Output shape:", first_example['output'].shape)